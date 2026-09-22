"""SYNTHETIC NOT-LAW fixtures for snake-case dataset contracts; never dataset content."""

from copy import deepcopy
from hashlib import sha256

import pytest
from pydantic import ValidationError

from taxcalcbench.schema import (
    CITATION_HEADERS,
    QUESTION_HEADERS,
    CitationRow,
    FamilyDraft,
    GoldStep,
    QuestionRow,
    ReviewItem,
    normalize_and_check_family,
)

TEXT = "SYNTHETIC TEST ONLY, NOT LAW: the synthetic rate is 0.10. No real tax is described."


def family_payload(family_id="A-F01", question_numbers=None, *, country="ZZ", language="ru", currency="TEST"):
    """A reusable fake family for offline orchestration tests, not tax examples."""
    numbers = question_numbers or list(range(1, 6))
    variants = ["original", "threshold", "eligibility", "numbers", "temporal"]
    scenarios = ["original baseline", "threshold condition", "registration condition",
                 "expense allocation", "transitional credit"]
    questions, citations = [], []
    for number, variant, scenario in zip(numbers, variants, scenarios, strict=True):
        questions.append({
            "no": number, "family": family_id, "variant": variant, "country": country, "language": language,
            "tax_category": "Income, profits and capital gains", "primary_taxpayer": "Company / corporation",
            "year_period_target": "2020-2026", "case_year": 2025, "difficulty": "D3",
            "question": f"SYNTHETIC NOT-LAW: test {scenario} with base 100 {currency}.",
            "final_answer_text": f"SYNTHETIC fixture answer: 10 {currency}", "answer_value": "10", "unit": currency,
            "gold_steps": [
                {"step": "Read the synthetic rate.", "result": "0.10", "unit": "1", "citations": ["C1"]},
                {"step": f"Multiply the test {scenario} base by the synthetic rate.", "result": "10",
                 "unit": currency, "citations": ["C1"]},
            ],
            "status": "draft", "note": None if variant == "original" else f"SYNTHETIC change: {scenario}.",
        })
        citations.append({
            "question_no": number, "language": language, "citation_id": "C1",
            "official_title": "SYNTHETIC TEST ONLY, NOT LAW", "law_number": None,
            "article_section": "Synthetic test clause", "paragraph": None, "subparagraph": None,
            "rule_applies_from": "2025-01-01", "rule_applies_until": "2025-12-31",
            "official_source_url": None, "source_file": "toy-source", "supporting_passage": "the synthetic rate is 0.10",
        })
    return {"topic": "SYNTHETIC schema fixture, not a benchmark family", "questions": questions, "citations": citations}


def fixture_family(tmp_path, **kwargs):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "original.txt").write_text(TEXT, encoding="utf-8")
    (source_dir / "readable.txt").write_text(TEXT, encoding="utf-8")
    draft = FamilyDraft.model_validate(family_payload(**kwargs))
    country, language, currency = draft.questions[0].country, draft.questions[0].language, draft.questions[0].unit
    config = {"country": country, "language": language, "currency": currency,
              "generation": {"periods": [{"label": "2020-2026", "start": 2020, "end": 2026, "cases": 75}]}}
    inventory = {"documents": [{"id": "toy-source", "title": "SYNTHETIC TEST ONLY, NOT LAW",
        "url": "https://synthetic.source.invalid/toy", "final_url": "https://synthetic.source.invalid/toy/final",
        "raw_path": "original.txt", "text_path": "readable.txt", "sha256": sha256(TEXT.encode()).hexdigest(),
        "text_sha256": sha256(TEXT.encode()).hexdigest(), "language": language, "country": country, "kind": "law"}]}
    return draft, config, inventory, source_dir


@pytest.fixture
def setup(tmp_path):
    return fixture_family(tmp_path)


def check(setup):
    draft, config, inventory, source_dir = setup
    return normalize_and_check_family(draft, config, inventory, source_dir,
                                      family_id="A-F01", question_numbers=list(range(1, 6)))


def test_exact_snake_case_export_headers_and_required_null_fields(setup):
    draft, failures = check(setup)
    assert not failures
    assert list(draft.questions[0].model_dump(by_alias=True)) == QUESTION_HEADERS
    assert list(draft.citations[0].model_dump(by_alias=True)) == CITATION_HEADERS
    assert draft.questions[0].model_dump(by_alias=True)["note"] is None
    assert draft.citations[0].source_file == "original.txt"
    assert draft.citations[0].official_source_url == "https://synthetic.source.invalid/toy/final"
    assert setup[0].citations[0].source_file == "toy-source"


def test_unknown_unit_error_lists_configured_currency_and_accepted_values(setup):
    draft, config, _, _ = setup
    draft.questions[0].unit = "SYNTHETIC translated currency name"
    draft.questions[0].gold_steps[-1].unit = draft.questions[0].unit
    _, failures = check(setup)
    error = next(issue for issue in failures if "unit is not a workbook/configured unit" in issue)
    assert "allowed values:" in error
    assert config["currency"] in error and "percent" in error and "days" in error


@pytest.mark.parametrize("field,value", [("question_id", "extra"), ("confidence", 1)])
def test_added_question_fields_are_rejected(field, value):
    payload = family_payload()["questions"][0] | {field: value}
    with pytest.raises(ValidationError, match="Extra inputs"):
        QuestionRow.model_validate(payload)


def test_missing_optional_column_is_not_silently_omitted():
    payload = family_payload()["questions"][0]
    payload.pop("note")
    with pytest.raises(ValidationError, match="Field required"):
        QuestionRow.model_validate(payload)


@pytest.mark.parametrize("field,value", [("no", True), ("case_year", 2025.0), ("tax_category", "1000"),
    ("difficulty", "D1"), ("primary_taxpayer", "Corporation"), ("variant", "Variant 1"),
    ("answer_value", "1,000.00"), ("answer_value", 10), ("answer_value", "NaN")])
def test_wrong_lookup_values_or_inexact_numeric_types_reject(field, value):
    with pytest.raises(ValidationError):
        QuestionRow.model_validate(family_payload()["questions"][0] | {field: value})


@pytest.mark.parametrize("start,end", [("2025-02-30", None), ("2025/01/01", None), ("2025-01-02", "2025-01-01")])
def test_invalid_applicability_dates_reject(start, end):
    payload = family_payload()["citations"][0] | {"rule_applies_from": start, "rule_applies_until": end}
    with pytest.raises(ValidationError):
        CitationRow.model_validate(payload)


@pytest.mark.parametrize("problem,expected", [
    ("count", "assigned question"), ("duplicate_number", "duplicate question"), ("original", "exactly one original"),
    ("country", "assignment"), ("language", "assignment"), ("family", "assignment"),
    ("year", "target period"), ("status", "cannot assign validated"),
    ("note", "change explanation"), ("source", "inventory source"), ("orphan", "does not join"),
    ("duplicate_citation", "duplicate citation"), ("formula", "spreadsheet formula"),
])
def test_immediate_family_checks(setup, problem, expected):
    draft = setup[0]
    if problem == "count":
        draft.questions.pop()
    elif problem == "duplicate_number":
        draft.questions[1].no = 1
    elif problem == "original":
        draft.questions[1].variant = "original"
    elif problem in {"country", "language", "family"}:
        setattr(draft.questions[0], problem, {"country": "XX", "language": "de", "family": "B-F01"}[problem])
    elif problem == "year":
        draft.questions[0].case_year = 2015
    elif problem == "status":
        draft.questions[0].status = "validated"
    elif problem == "note":
        draft.questions[1].note = None
    elif problem == "source":
        draft.citations[0].source_file = "unknown"
    elif problem == "orphan":
        draft.citations[0].question_no = 99
    elif problem == "duplicate_citation":
        draft.citations.append(deepcopy(draft.citations[0]))
    else:
        draft.questions[0].question = "=Options!B2"
    _, failures = check(setup)
    assert any(expected in failure for failure in failures)


@pytest.mark.parametrize("label", ["2025", "2020–2026", "2010-2019"])
def test_period_label_is_derived_from_the_checked_case_year(setup, label):
    setup[0].questions[0].year_period_target = label
    draft, failures = check(setup)
    assert not failures
    assert draft.questions[0].case_year == 2025
    assert draft.questions[0].year_period_target == "2020-2026"


@pytest.mark.parametrize("index,field,value,expected", [
    (1, "citations", ["C9"], "missing citation"),
    (1, "result", "11", "differs"), (1, "unit", "USD", "differs"),
    (1, "unit", None, "differs"),
])
def test_gold_steps_citation_join_and_final_answer(setup, index, field, value, expected):
    setattr(setup[0].questions[0].gold_steps[index], field, value)
    _, failures = check(setup)
    assert any(expected in failure for failure in failures)


@pytest.mark.parametrize("change", [{"citations": []}, {"citations": ["invalid"]}, {"step": ""}, {"result": ""}])
def test_gold_step_objects_require_complete_fields(setup, change):
    payload = setup[0].questions[0].gold_steps[0].model_dump() | change
    with pytest.raises(ValidationError):
        GoldStep.model_validate(payload)


def test_dimensionless_intermediate_share_accepts_null_unit_but_final_units_remain_required(setup):
    question = setup[0].questions[0]
    question.gold_steps[0].step = "Determine the synthetic dimensionless ownership share."
    question.gold_steps[0].result = "0.2"
    question.gold_steps[0].unit = None
    assert not check(setup)[1]
    question.gold_steps[-1].unit = None
    assert any("Final gold_steps result/unit differs" in issue for issue in check(setup)[1])
    with pytest.raises(ValidationError, match="numeric answer requires a unit"):
        QuestionRow.model_validate(question.model_dump() | {"unit": None})


def test_non_numeric_steps_can_have_null_unit(setup):
    q = setup[0].questions[0]
    q.gold_steps[0].result = "Synthetic eligibility established"
    q.gold_steps[0].unit = None
    assert not check(setup)[1]


def test_source_url_and_file_are_bound_not_taken_from_model(setup):
    setup[0].citations[0].official_source_url = "https://invented.invalid/fake"
    bound, failures = check(setup)
    assert not failures and bound.citations[0].official_source_url == setup[2]["documents"][0]["final_url"]
    assert not check((bound, *setup[1:]))[1]


def replace_source(setup, text):
    _, _, inventory, folder = setup
    (folder / "original.txt").write_text(text, encoding="utf-8")
    (folder / "readable.txt").write_text(text, encoding="utf-8")
    inventory["documents"][0].update(sha256=sha256(text.encode()).hexdigest(), text_sha256=sha256(text.encode()).hexdigest())


def test_whitespace_binding_restores_exact_source_without_altering_original(setup):
    replace_source(setup, TEXT.replace("synthetic rate", "synthetic\u00a0\nrate"))
    original = deepcopy(setup[0].model_dump(by_alias=True))
    bound, failures = check(setup)
    assert not failures and "\u00a0\n" in bound.citations[0].supporting_passage
    assert setup[0].model_dump(by_alias=True) == original


@pytest.mark.parametrize("problem", ["ambiguous", "changed_number", "changed_word", "punctuation", "stitched", "unavailable", "unavailable_whitespace"])
def test_only_unique_contiguous_whitespace_quote_matches_are_allowed(setup, problem):
    quote = setup[0].citations[0].supporting_passage
    if problem == "ambiguous":
        replace_source(setup, quote.replace(" ", "\n") + "\nOTHER\n" + quote.replace(" ", "\t"))
    elif problem == "changed_number":
        setup[0].citations[0].supporting_passage = quote.replace("0.10", "0.20")
    elif problem == "changed_word":
        setup[0].citations[0].supporting_passage = quote.replace("rate", "base")
    elif problem == "punctuation":
        setup[0].citations[0].supporting_passage = quote.replace("is", "is:")
    elif problem == "stitched":
        replace_source(setup, quote.replace("rate is", "rate [intervening material] is"))
    else:
        marker = "[UNAVAILABLE PROVISION: synthetic formula]"
        replace_source(setup, TEXT + marker)
        setup[0].citations[0].supporting_passage = marker.replace(" ", "\t") if problem == "unavailable_whitespace" else marker
    assert check(setup)[1]


@pytest.mark.parametrize("field,value", [("language", "de"), ("country", "XX"), ("kind", "navigation")])
def test_source_policy_metadata_is_enforced(setup, field, value):
    setup[2]["documents"][0][field] = value
    assert check(setup)[1]


def test_number_only_rewording_does_not_create_a_substantive_variant(setup):
    original, changed = setup[0].questions[:2]
    changed.question = original.question.replace("100", "200")
    changed.gold_steps = deepcopy(original.gold_steps)
    assert any("only numeric/date" in failure for failure in check(setup)[1])


def test_passed_review_requires_recalculation_and_no_open_issues():
    good = {"question_no": 1, "passed": True, "issues": [], "recomputed_answer": "10",
            "unit": "TEST", "calculation": "100 * 0.10"}
    assert ReviewItem.model_validate(good).passed
    for change in [{"calculation": None}, {"recomputed_answer": None}, {"unit": None}]:
        with pytest.raises(ValidationError):
            ReviewItem.model_validate(good | change)
    assert not ReviewItem.model_validate({"question_no": 1, "passed": False, "issues": ["SYNTHETIC missing input"],
        "recomputed_answer": None, "unit": None, "calculation": None}).passed
    contradictory = ReviewItem.model_validate(good | {"issues": ["Unresolved"]})
    assert not contradictory.passed and contradictory.issues == ["Unresolved"]
    assert contradictory.calculation == good["calculation"]

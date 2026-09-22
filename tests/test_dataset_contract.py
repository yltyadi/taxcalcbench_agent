"""Synthetic-only contracts for configurable temporal and missing-information cases."""

from copy import deepcopy
from hashlib import sha256

import pytest
from pydantic import ValidationError

from taxcalcbench.schema import (
    CITATION_HEADERS,
    QUESTION_HEADERS,
    FamilyDraft,
    FamilyPlan,
    GoldStep,
    QuestionRow,
    ReviewItem,
    ReviewReport,
    normalize_and_check_family,
)

TEXT = "SYNTHETIC TEST ONLY, NOT LAW: a synthetic multiplier depends on the specified class."


def question(no=1, variant="original"):
    return {
        "no": no, "family": "test-family", "variant": variant, "country": "ZZ", "language": "en",
        "tax_category": "Income, profits and capital gains", "primary_taxpayer": "Company / corporation",
        "year_period_target": "test", "case_year": 2025, "difficulty": "D2",
        "question": f"SYNTHETIC NOT-LAW scenario {no}: apply the stated rule for 2025 to 100 TEST.",
        "final_answer_text": "SYNTHETIC NOT-LAW: the computed result is 10 TEST.", "answer_value": "10",
        "unit": "TEST", "gold_steps": [{"step": "Apply the synthetic rule.", "result": "10",
            "unit": "TEST", "citations": ["C1"]}], "status": "draft",
        "note": None if variant == "original" else "SYNTHETIC scenario change.",
    }


def missing_question(no=2):
    return question(no, "missing_information") | {
        "question": "SYNTHETIC NOT-LAW: compute the amount for a 100 TEST base without a specified class.",
        "answer_value": "insufficient information", "unit": None,
        "final_answer_text": "The synthetic class is omitted. It determines the multiplier, so no unique answer follows.",
        "gold_steps": [{"step": "The omitted synthetic class determines the required multiplier.",
            "result": "insufficient information", "unit": None, "citations": ["C1"]}],
    }


def citation(no):
    return {"question_no": no, "language": "en", "citation_id": "C1", "official_title": "SYNTHETIC NOT-LAW",
            "law_number": None, "article_section": "Test clause", "paragraph": None, "subparagraph": None,
            "rule_applies_from": "2020-01-01", "rule_applies_until": None, "official_source_url": None,
            "source_file": "synthetic-source", "supporting_passage": "a synthetic multiplier depends on the specified class"}


def plan_case(no=1, variant="original", year=2025, derived_from=None, missing_fact=None):
    return {"question_no": no, "variant": variant, "case_year": year,
            "tax_category": "Income, profits and capital gains", "primary_taxpayer": "Company / corporation",
            "scenario_outline": "SYNTHETIC NOT-LAW scenario.", "legal_difference": "SYNTHETIC rule interaction.",
            "legal_assumptions": ["SYNTHETIC assumptions."],
            "evidence": [{"source_id": "synthetic-source", "start_line": 1, "end_line": 1,
                          "purpose": "SYNTHETIC rule."}], "derived_from": derived_from, "missing_fact": missing_fact}


def normalize(tmp_path, rows):
    source = tmp_path / "law.txt"
    source.write_text(TEXT, encoding="utf-8")
    digest = sha256(TEXT.encode()).hexdigest()
    inventory = {"documents": [{"id": "synthetic-source", "country": "ZZ", "language": "en", "kind": "law",
        "title": "SYNTHETIC NOT-LAW", "raw_path": source.name, "text_path": source.name,
        "url": "https://synthetic.invalid/law", "sha256": digest, "text_sha256": digest}]}
    config = {"country": "ZZ", "language": "en", "currency": "TEST",
              "generation": {"periods": [{"label": "test", "start": 2000, "end": 2030}]}}
    draft = FamilyDraft(topic="SYNTHETIC NOT-LAW", questions=rows, citations=[citation(row["no"]) for row in rows])
    return normalize_and_check_family(draft, config, inventory, tmp_path,
                                      family_id="test-family", question_numbers=[row["no"] for row in rows])


def test_exact_snake_case_headers_and_nested_steps():
    row = QuestionRow.model_validate(question()).model_dump()
    assert list(row) == QUESTION_HEADERS
    assert list(row["gold_steps"][0]) == ["step", "result", "unit", "citations"]
    assert all(field == field.lower() and " " not in field for field in QUESTION_HEADERS + CITATION_HEADERS)


@pytest.mark.parametrize("old_key", ["No", "Answer Value", "Gold Steps", "year_period"])
def test_unsupported_question_fields_are_rejected(old_key):
    with pytest.raises(ValidationError, match="Extra inputs"):
        QuestionRow.model_validate(question() | {old_key: "unsupported"})


@pytest.mark.parametrize("steps", ["Step 1: unsupported format", [], [{"step": "x", "result": "1", "unit": "TEST", "citations": []}]])
def test_gold_steps_require_a_nonempty_array_of_cited_objects(steps):
    with pytest.raises(ValidationError):
        QuestionRow.model_validate(question() | {"gold_steps": steps})


def test_step_keys_are_exact_and_citation_ids_are_distinct():
    data = question()["gold_steps"][0]
    for change in [{"number": 1}, {"citations": ["C1", "C1"]}, {"citations": ["Citation 1"]}]:
        with pytest.raises(ValidationError):
            GoldStep.model_validate(data | change)


@pytest.mark.parametrize("count", [1, 2, 6, 9])
def test_family_cardinality_comes_from_assignment(tmp_path, count):
    rows = [question()]
    for no in range(2, count + 1):
        row = question(no, "eligibility")
        row["question"] = "SYNTHETIC scenario " + ("x" * no)
        rows.append(row)
    result, failures = normalize(tmp_path, rows)
    assert not failures
    assert len(result.questions) == count
    assert all(c.source_file == "law.txt" for c in result.citations)


def test_temporal_number_only_similarity_is_deliberate(tmp_path):
    original = question()
    temporal = deepcopy(original) | {"no": 2, "variant": "temporal", "case_year": 2024, "note": "Change only the year."}
    temporal["question"] = original["question"].replace("2025", "2024")
    assert not normalize(tmp_path, [original, temporal])[1]
    temporal["variant"] = "numbers"
    assert any("only numeric/date" in issue for issue in normalize(tmp_path, [original, temporal])[1])


def test_missing_information_export_has_no_invented_numeric_answer(tmp_path):
    draft, failures = normalize(tmp_path, [question(), missing_question()])
    assert not failures
    missing = draft.questions[1]
    assert missing.answer_value == missing.gold_steps[-1].result == "insufficient information"
    assert missing.unit is missing.gold_steps[-1].unit is None
    assert len(missing.model_dump()) == 17


@pytest.mark.parametrize("change", [
    {"answer_value": "10"}, {"answer_value": "Insufficient information"}, {"unit": "TEST"},
    {"final_answer_text": "insufficient information"},
    {"gold_steps": [{"step": "An input is absent.", "result": "10", "unit": None, "citations": ["C1"]}]},
])
def test_missing_information_contract_rejects_pretend_answers(change):
    with pytest.raises(ValidationError):
        QuestionRow.model_validate(missing_question() | change)


def test_numeric_case_cannot_claim_insufficient_information():
    with pytest.raises(ValidationError):
        QuestionRow.model_validate(question() | {"answer_value": "insufficient information"})
    with pytest.raises(ValidationError):
        QuestionRow.model_validate(question() | {"unit": None})


def test_plan_allows_variable_sizes_and_valid_temporal_missing_dependencies():
    for cases in [[plan_case()], [plan_case(), plan_case(2, "temporal", 2024, 1),
                                  plan_case(3, "missing_information", 2025, 1, "Omitted synthetic class.")]]:
        assert len(FamilyPlan(family_id="test", topic="Synthetic", cases=cases).cases) == len(cases)


@pytest.mark.parametrize("case", [
    plan_case(2, "temporal", 2024), plan_case(2, "temporal", 2025, 1),
    plan_case(2, "temporal", 2024, 3), plan_case(2, "missing_information", 2025, 1),
    plan_case(2, "missing_information", 2024, 1, "Omitted class."),
    plan_case(2, "eligibility", 2025, missing_fact="Should not exist."),
])
def test_plan_rejects_unusable_dependencies(case):
    with pytest.raises(ValidationError):
        FamilyPlan(family_id="test", topic="Synthetic", cases=[plan_case(), case])


def test_regular_review_is_numeric_and_ignores_nonexported_metadata():
    numeric = {"question_no": 1, "passed": True, "issues": [], "recomputed_answer": "10",
               "unit": "TEST", "calculation": "100*0.1"}
    assert ReviewItem.model_validate(numeric).passed
    assert ReviewItem.model_validate(numeric | {"missing_fact": None, "alternatives": []}).model_dump() == numeric
    for change in [{"recomputed_answer": "insufficient information"}, {"calculation": None}, {"unit": None}]:
        with pytest.raises(ValidationError):
            ReviewItem.model_validate(numeric | change)
    assert len(ReviewReport(family_id="test", reviewer="B", questions=[numeric | {"question_no": i} for i in range(1, 9)],
                            family_issues=[]).questions) == 8


@pytest.mark.parametrize("end", [None, "2025-12-31"])
def test_unknown_start_bound_remains_null_without_claiming_case_applicability(end):
    from taxcalcbench.schema import CitationRow

    row = CitationRow.model_validate(citation(1) | {"rule_applies_from": None, "rule_applies_until": end})
    assert row.rule_applies_from is None and row.rule_applies_until == end
    with pytest.raises(ValidationError, match="precedes"):
        CitationRow.model_validate(citation(1) | {"rule_applies_from": "2026-01-01", "rule_applies_until": "2025-12-31"})

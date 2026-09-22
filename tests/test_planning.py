"""SYNTHETIC NOT-LAW fixtures for the internal planner/writer handoff."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_schema import family_payload

from taxcalcbench.schema import (
    CITATION_HEADERS,
    QUESTION_HEADERS,
    CitationRow,
    FamilyPlan,
    QuestionDraft,
    QuestionRow,
)


def plan_payload():
    return {
        "family_id": "A-F01", "topic": "SYNTHETIC planning fixture, not real law",
        "cases": [{
            "question_no": number, "variant": "original" if number == 1 else "eligibility", "case_year": 2025,
            "tax_category": "Income, profits and capital gains", "primary_taxpayer": "Company / corporation",
            "scenario_outline": f"SYNTHETIC scenario {number}, not a tax question.",
            "legal_difference": f"SYNTHETIC changed condition {number}.",
            "legal_assumptions": ["SYNTHETIC assumptions only, no actual law."],
            "evidence": [{"source_id": "toy-source", "start_line": 1, "end_line": 3,
                          "purpose": "SYNTHETIC supporting test text."}],
        } for number in range(1, 6)],
    }


def test_plan_has_five_cases_and_can_repeat_supported_variant_labels():
    plan = FamilyPlan.model_validate(plan_payload())
    assert len(plan.cases) == 5
    assert plan.cases[-1].variant == "eligibility"
    assert "questions" not in plan.model_dump()
    assert "citations" not in plan.model_dump()


@pytest.mark.parametrize("change", ["empty", "duplicate", "original_later", "two_originals", "answer"])
def test_malformed_family_plan_rejects(change):
    payload = deepcopy(plan_payload())
    if change == "empty":
        payload["cases"].clear()
    elif change == "duplicate":
        payload["cases"][1]["question_no"] = 1
    elif change == "original_later":
        payload["cases"].reverse()
    elif change == "two_originals":
        payload["cases"][1]["variant"] = "original"
    else:
        payload["cases"][0]["answer"] = "10"
    with pytest.raises(ValidationError):
        FamilyPlan.model_validate(payload)


@pytest.mark.parametrize("start,end", [(0, 3), (5, 4), (1, 401), (True, 3), (1, 3.0)])
def test_unreadable_evidence_ranges_reject(start, end):
    payload = plan_payload()
    payload["cases"][0]["evidence"][0].update(start_line=start, end_line=end)
    with pytest.raises(ValidationError):
        FamilyPlan.model_validate(payload)


def test_internal_planning_does_not_change_export_columns():
    assert list(QuestionRow.model_fields) == QUESTION_HEADERS
    assert list(CitationRow.model_fields) == CITATION_HEADERS


def test_single_question_handoff_has_exact_snake_case_rows():
    family = family_payload()
    draft = QuestionDraft.model_validate({"question": family["questions"][0], "citations": [family["citations"][0]]})
    assert draft.question.no == draft.citations[0].question_no == 1
    assert list(draft.question.model_dump(by_alias=True)) == QUESTION_HEADERS
    assert list(draft.citations[0].model_dump(by_alias=True)) == CITATION_HEADERS


@pytest.mark.parametrize("change", ["number", "language", "empty"])
def test_single_question_handoff_rejects_missing_or_unrelated_citations(change):
    family = family_payload()
    payload = {"question": family["questions"][0], "citations": [family["citations"][0]]}
    if change == "number":
        payload["citations"][0]["question_no"] = 2
    elif change == "language":
        payload["citations"][0]["language"] = "en"
    else:
        payload["citations"] = []
    with pytest.raises(ValidationError):
        QuestionDraft.model_validate(payload)

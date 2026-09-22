"""SYNTHETIC saved plans keep topics distinct across resumed families."""

from test_experts import ASSIGNMENT, plan_payload
from test_experts import setup as setup

from taxcalcbench.experts import ExpertRunner
from taxcalcbench.schema import FamilyPlan


def assignment(index):
    start = (index - 1) * 5 + 1
    return {"family_id": f"A-F{index:02d}", "question_numbers": list(range(start, start + 5)),
            "regular_question_numbers": list(range(start, start + 5)), "missing_question_numbers": [],
            "period_slots": [dict(slot, question_no=slot["question_no"] + start - 1) for slot in ASSIGNMENT["period_slots"]],
            "category": "Income, profits and capital gains"}


def synthetic_plan(assigned):
    payload = plan_payload()
    payload.update(family_id=assigned["family_id"], topic="SYNTHETIC topic " + assigned["family_id"])
    for case, number in zip(payload["cases"], assigned["question_numbers"], strict=True):
        case["question_no"] = number
        if case["derived_from"] is not None:
            case["derived_from"] = assigned["question_numbers"][0]
    return FamilyPlan.model_validate(payload)


async def test_restarted_planner_reads_saved_topics_and_merges_completed_family_summaries(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    directory = setup[3] / "plans"
    directory.mkdir()
    plan = synthetic_plan(assignment(1))
    (directory / "A-F01.json").write_text(plan.model_dump_json())
    seen = []

    async def invoke(task, expert, payload, output_type, tools):
        seen.extend(payload["previous_topics"])
        return synthetic_plan(payload["assignment"])

    monkeypatch.setattr(runner, "_invoke", invoke)
    await runner.plan("A", assignment(3), [
        {"family_id": "A-F01", "topic": "SYNTHETIC older incomplete summary"},
        {"family_id": "A-F02", "topic": "SYNTHETIC completed family without saved plan"},
    ])
    assert [item["family_id"] for item in seen] == ["A-F01", "A-F02"]
    assert seen[0]["topic"] == plan.topic and len(seen[0]["variants"]) == 5
    assert seen[1]["topic"] == "SYNTHETIC completed family without saved plan"

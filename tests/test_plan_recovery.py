"""Resume synthetic planner output without repeating official-source research."""

import json
from copy import deepcopy

import pytest
from test_experts import ASSIGNMENT, plan_payload
from test_experts import setup as setup

from taxcalcbench.experts import ExpertRunError, ExpertRunner
from taxcalcbench.schema import FamilyPlan


async def failed_plan(runner, monkeypatch, *, valid=False):
    saved = {}

    async def invoke(task, expert, payload, output_type, tools):
        saved.update(payload=deepcopy(payload))
        operation = runner.work_dir / "operations/plan-A-synthetic"
        operation.mkdir(parents=True)
        (operation / "input.json").write_text(json.dumps({
            "task": task, "expert": expert, "model": runner.model, "input": payload}))
        draft = plan_payload()
        if not valid:
            draft["cases"][1]["derived_from"] = None
        text = json.dumps(draft)
        if valid:
            (operation / "result.json").write_text(json.dumps({"output": draft}))
        else:
            (operation / "invalid_output.json").write_text(json.dumps({"output_text": text}))
        raise ExpertRunError("SYNTHETIC output interruption")

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises(ExpertRunError, match="SYNTHETIC output interruption"):
        await runner.plan("A", ASSIGNMENT, [])
    return saved["payload"]


async def test_resume_invalid_plan_corrects_saved_output_without_research(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    await failed_plan(runner, monkeypatch)
    calls = []

    async def repair(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert kwargs == {"repair_only": True}
        assert payload["validation_errors"][0]["path"] == "cases[1]"
        assert "require derived_from" in payload["validation_errors"][0]["message"]
        assert json.loads(payload["previous_output"])["cases"][1]["derived_from"] is None
        return FamilyPlan.model_validate(plan_payload())

    monkeypatch.setattr(runner, "_invoke", repair)
    plan = await runner.plan("A", ASSIGNMENT, [])
    assert plan.cases[1].derived_from == 1 and calls == ["plan"]
    assert await runner.plan("A", ASSIGNMENT, []) == plan
    assert calls == ["plan"]


async def test_resume_finished_plan_response_uses_no_model_calls(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    await failed_plan(runner, monkeypatch, valid=True)

    async def unexpected(*args, **kwargs):
        pytest.fail("Completed output must not call the model again")

    monkeypatch.setattr(runner, "_invoke", unexpected)
    result = await runner.plan("A", ASSIGNMENT, [])
    assert result.family_id == ASSIGNMENT["family_id"]


@pytest.mark.parametrize("change", ["source", "assignment", "language", "model", "topics"])
async def test_recovery_requires_matching_source_and_task_identity(setup, monkeypatch, change):
    runner = ExpertRunner(*setup, {})
    payload = await failed_plan(runner, monkeypatch)
    assert runner._saved_output("plan", "A", payload) is not None
    if change == "source":
        payload["sources"][0]["sha256"] = "SYNTHETIC changed evidence"
    elif change == "assignment":
        payload["assignment"]["family_id"] = "A-F02"
    elif change == "language":
        payload["language"] = "fr"
    elif change == "model":
        runner.model = "SYNTHETIC different model"
    else:
        payload["previous_topics"] = [{"family_id": "A-F02", "topic": "SYNTHETIC completed topic"}]
    assert runner._saved_output("plan", "A", payload) is None

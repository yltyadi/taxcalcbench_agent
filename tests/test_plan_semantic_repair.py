"""One targeted local-plan correction preserves completed planning work."""

import json
from copy import deepcopy

import pytest
from test_experts import ASSIGNMENT, plan_payload, question_payload
from test_experts import setup as setup
from test_plan_recovery import failed_plan

from taxcalcbench.experts import ExpertRunError, ExpertRunner, _SourceGap
from taxcalcbench.schema import FamilyPlan


def wrong_period_plan():
    result = plan_payload()
    result["cases"][1]["case_year"] = 2024
    return result


async def test_wrong_period_plan_gets_one_targeted_repair_before_any_writer(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    bad = wrong_period_plan()
    plans, written = [], []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        if task == "plan":
            plans.append(payload)
            if len(plans) == 1:
                return FamilyPlan.model_validate(bad)
            assert len(plans) == 2 and not written
            assert payload["previous_plan"]["cases"][1]["case_year"] == 2024
            assert payload["assignment"] == ASSIGNMENT
            assert any("Question 2" in issue and "2010-2019" in issue for issue in payload["feedback"])
            assert {row["source_id"] for row in payload["source_excerpts"]} == {"s_toy"}
            assert not kwargs.get("repair_only")
            return FamilyPlan.model_validate(plan_payload())
        assert task == "create" and len(plans) == 2
        assert all(case["case_year"] == plan_payload()["cases"][index]["case_year"]
                   for index, case in enumerate(payload["family_plan"]["cases"]))
        number = payload["case"]["question_no"]
        written.append(number)
        tools.calculate("2+3")
        return output_type.model_validate(question_payload(number))

    monkeypatch.setattr(runner, "_invoke", invoke)
    family = await runner.create("A", ASSIGNMENT, [])
    assert written == [1, 2, 3, 4, 5] and family.questions[1].case_year == 2015
    assert bad["cases"][1]["case_year"] == 2024  # Python does not rewrite assigned years.
    saved = json.loads((setup[3] / "plans/A-F01.json").read_text())
    assert saved["cases"][1]["case_year"] == 2015


async def test_repeated_invalid_period_stops_after_one_repair_without_publishing_plan(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, *args, **kwargs):
        assert task == "plan"
        calls.append(task)
        assert len(calls) <= 2
        return FamilyPlan.model_validate(wrong_period_plan())

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises(ExpertRunError, match="Question 2.*assigned period.*2010-2019"):
        await runner.create("A", ASSIGNMENT, [])
    assert len(calls) == 2 and not (setup[3] / "plans/A-F01.json").exists()
    assert not (setup[3] / "questions").exists()


async def test_resume_after_invalid_cached_semantic_repair_gets_one_fresh_attempt(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    operations = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        operations.append(deepcopy(payload))
        assert task == "plan"
        output = wrong_period_plan() if len(operations) <= 2 else plan_payload()
        if len(operations) == 3:
            assert payload["previous_plan"]["cases"][1]["case_year"] == 2024
            assert any("assigned period" in issue for issue in payload["feedback"])
        directory = setup[3] / f"operations/plan-A-synthetic-{len(operations)}"
        directory.mkdir(parents=True)
        (directory / "input.json").write_text(json.dumps({"model": runner.model, "input": payload}))
        (directory / "result.json").write_text(json.dumps({"output": output}))
        return FamilyPlan.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises(ExpertRunError, match="assigned period"):
        await runner.plan("A", ASSIGNMENT, [])
    assert len(operations) == 2
    assert (await runner.plan("A", ASSIGNMENT, [])).cases[1].case_year == 2015
    assert len(operations) == 3
    await runner.plan("A", ASSIGNMENT, [])
    assert len(operations) == 3


async def test_saved_wrong_period_plan_is_repaired_after_additive_source_growth(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    original_payload = await failed_plan(runner, monkeypatch, valid=True)
    path = setup[3] / "operations/plan-A-synthetic/result.json"
    path.write_text(json.dumps({"output": wrong_period_plan()}))
    addition = deepcopy(setup[1]["documents"][0]) | {"id": "s_new", "url": "https://synthetic.invalid/new"}
    runner.inventory["documents"].append(addition)
    current_payload = original_payload | {"sources": runner._inventory_summary()}
    assert runner._saved_output("plan", "A", current_payload) is not None
    calls = []

    async def repair(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert payload["previous_plan"]["cases"][1]["case_year"] == 2024
        assert len(payload["sources"]) == 2
        assert any("Question 2" in issue for issue in payload["feedback"])
        return FamilyPlan.model_validate(plan_payload())

    monkeypatch.setattr(runner, "_invoke", repair)
    assert (await runner.plan("A", ASSIGNMENT, [])).cases[1].case_year == 2015
    assert calls == ["plan"]


@pytest.mark.parametrize("change", ["sha256", "text_sha256", "country", "remove"])
async def test_saved_plan_additive_match_rejects_changed_or_removed_old_source(setup, monkeypatch, change):
    runner = ExpertRunner(*setup, {})
    payload = await failed_plan(runner, monkeypatch, valid=True)
    payload["sources"].append(deepcopy(payload["sources"][0]) | {"id": "s_new"})
    if change == "remove":
        payload["sources"].pop(0)
    else:
        payload["sources"][0][change] = "SYNTHETIC changed source identity"
    assert runner._saved_output("plan", "A", payload) is None


async def test_invalid_evidence_is_identified_and_excluded_from_repair_context(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    bad = plan_payload()
    bad["cases"][0]["evidence"].append({"source_id": "s_unknown", "start_line": 2,
        "end_line": 4, "purpose": "SYNTHETIC invalid pointer"})
    calls = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert task == "plan"
        if len(calls) == 1:
            return FamilyPlan.model_validate(bad)
        assert len(calls) == 2
        assert any("s_unknown lines 2-4" in issue for issue in payload["feedback"])
        assert {row["source_id"] for row in payload["source_excerpts"]} == {"s_toy"}
        return FamilyPlan.model_validate(plan_payload())

    monkeypatch.setattr(runner, "_invoke", invoke)
    assert (await runner.plan("A", ASSIGNMENT, [])).family_id == "A-F01"
    assert len(calls) == 2


@pytest.mark.parametrize("repair_error", [ExpertRunError("SYNTHETIC provider failure"), _SourceGap("SOURCE_GAP: SYNTHETIC missing law", stage="plan")])
async def test_semantic_repair_never_retries_provider_or_source_gap_failures(setup, monkeypatch, repair_error):
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, *args, **kwargs):
        calls.append(task)
        if len(calls) == 1:
            return FamilyPlan.model_validate(wrong_period_plan())
        assert len(calls) == 2
        raise repair_error

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises(type(repair_error), match="SYNTHETIC") as caught:
        await runner.plan("A", ASSIGNMENT, [])
    assert caught.value is repair_error and len(calls) == 2

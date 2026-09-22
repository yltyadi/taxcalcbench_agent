"""Declared statutory dependencies must reach writers with verified local evidence."""

import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_experts import ASSIGNMENT, plan_payload
from test_experts import setup as setup

from taxcalcbench import experts
from taxcalcbench.experts import ExpertRunError, ExpertRunner
from taxcalcbench.schema import FamilyPlan, LegalParameter


def add_parameter_source(setup):
    text = "SYNTHETIC statutory index\nSYNTHETIC index value: 7.\nSYNTHETIC commencement."
    for name in ("index.raw.txt", "index.txt"):
        (setup[2] / name).write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = deepcopy(setup[1]["documents"][0])
    document.update(id="s_index", title="SYNTHETIC index law", raw_path="index.raw.txt",
        text_path="index.txt", sha256=digest, text_sha256=digest, extraction_gaps=[],
        url="https://synthetic.invalid/index", final_url="https://synthetic.invalid/index")
    setup[1]["documents"].append(document)
    parameter = {"name": "SYNTHETIC statutory index", "value": "7",
        "evidence": [{"source_id": "s_index", "start_line": 1, "end_line": 3,
                      "purpose": "SYNTHETIC official index value"}]}
    return document, parameter


def test_incomplete_plan_schema_remains_readable_and_zero_parameter_requires_evidence():
    payload = plan_payload()
    for case in payload["cases"]:
        case.pop("legal_parameters")
    assert all(case.legal_parameters == [] for case in FamilyPlan.model_validate(payload).cases)
    span = payload["cases"][0]["evidence"][0]
    parameter = LegalParameter(name="SYNTHETIC exemption rate", value="0", evidence=[span])
    assert parameter.value == "0"
    with pytest.raises(ValidationError):
        LegalParameter(name="SYNTHETIC unsupported rate", value="5%", evidence=[])


def test_parameter_evidence_is_merged_and_duplicate_spans_read_once(setup):
    _, parameter = add_parameter_source(setup)
    payload = plan_payload()
    payload["cases"][0]["legal_parameters"].extend([parameter, deepcopy(parameter)])
    runner = ExpertRunner(*setup, {})
    excerpts = runner._evidence(FamilyPlan.model_validate(payload).cases)
    assert [item["source_id"] for item in excerpts] == ["s_toy", "s_index"]
    assert "index value: 7" in excerpts[1]["text"]


@pytest.mark.parametrize("failure", ["country", "language", "navigation", "raw_hash", "text_hash", "missing_lines", "gap"])
async def test_invalid_parameter_evidence_blocks_writer_even_when_case_evidence_is_valid(setup, monkeypatch, failure):
    document, parameter = add_parameter_source(setup)
    if failure in {"country", "language"}:
        document[failure] = "different"
    elif failure == "navigation":
        document["kind"] = "navigation"
    elif failure == "raw_hash":
        (setup[2] / document["raw_path"]).write_text("SYNTHETIC changed raw law")
    elif failure == "text_hash":
        (setup[2] / document["text_path"]).write_text("SYNTHETIC changed extracted law")
    elif failure == "missing_lines":
        parameter["evidence"][0]["end_line"] = 4
    else:
        document["extraction_gaps"] = [{"start": 3, "end": 30, "label": "SYNTHETIC unreadable provision"}]
    payload = plan_payload()
    payload["cases"][0]["legal_parameters"].append(parameter)
    experts._write(setup[3] / "plans/A-F01.json", payload)
    runner = ExpertRunner(*setup, {})

    async def unexpected(task, *args, **kwargs):
        assert task == "plan", "Unverified parameter evidence must not reach a writer"
        assert failure not in {"raw_hash", "text_hash"}, "Integrity failures must not trigger repair"
        return FamilyPlan.model_validate(payload)

    monkeypatch.setattr(runner, "_invoke", unexpected)
    with pytest.raises((ExpertRunError, ValueError)):
        await runner.create("A", ASSIGNMENT, [])
    assert not (setup[3] / "questions").exists()


async def test_incomplete_saved_plan_reopens_planning_and_preserves_existing_question_files(setup, monkeypatch):
    incomplete = plan_payload()
    for case in incomplete["cases"]:
        case.pop("legal_parameters")
    experts._write(setup[3] / "plans/A-F01.json", incomplete)
    checkpoint = setup[3] / "questions/A-F01/1.json"
    experts._write(checkpoint, {"synthetic": "existing completed question is outside planning"})
    original = checkpoint.read_bytes()
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, expert_id, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert not kwargs.get("repair_only")
        assert all(case["legal_parameters"] == [] for case in payload["previous_plan"]["cases"])
        assert len(payload["feedback"]) == 5 and all("legal_parameters" in issue for issue in payload["feedback"])
        assert payload["source_excerpts"][0]["source_id"] == "s_toy"
        return FamilyPlan.model_validate(plan_payload())

    monkeypatch.setattr(runner, "_invoke", invoke)
    plan = await runner.plan("A", ASSIGNMENT, [])
    assert all(case.legal_parameters for case in plan.cases)
    assert await runner.plan("A", ASSIGNMENT, []) == plan
    assert calls == ["plan"] and checkpoint.read_bytes() == original


async def test_finished_incomplete_model_response_reopens_dependency_research(setup, monkeypatch):
    incomplete = plan_payload()
    for case in incomplete["cases"]:
        case.pop("legal_parameters")
    runner = ExpertRunner(*setup, {})
    monkeypatch.setattr(runner, "_saved_output",
                        lambda task, expert, payload: None if "previous_plan" in payload else json.dumps(incomplete))
    calls = []

    async def invoke(task, expert_id, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert not kwargs.get("repair_only")
        assert "previous_plan" in payload and "source_excerpts" in payload
        assert all("legal_parameters" in issue for issue in payload["feedback"])
        return FamilyPlan.model_validate(plan_payload())

    monkeypatch.setattr(runner, "_invoke", invoke)
    assert all(case.legal_parameters for case in (await runner.plan("A", ASSIGNMENT, [])).cases)
    assert calls == ["plan"]


async def test_new_plan_without_parameters_is_rejected_before_saving_or_writing(setup, monkeypatch):
    payload = plan_payload()
    payload["cases"][2]["legal_parameters"] = []
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, *args, **kwargs):
        calls.append(task)
        return FamilyPlan.model_validate(payload)

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises(ExpertRunError, match="Question 3: declare all statutory calculation dependencies"):
        await runner.create("A", ASSIGNMENT, [])
    assert calls == ["plan", "plan"]
    assert not (setup[3] / "plans/A-F01.json").exists()
    assert not (setup[3] / "questions").exists()


async def test_writer_receives_parameter_values_and_their_separate_official_excerpts(setup, monkeypatch):
    _, parameter = add_parameter_source(setup)
    payload = plan_payload()
    payload["cases"][0]["legal_parameters"].append(parameter)
    experts._write(setup[3] / "plans/A-F01.json", payload)
    runner = ExpertRunner(*setup, {})

    async def inspect_writer(task, expert_id, payload, output_type, tools, **kwargs):
        assert task == "create"
        assert payload["case"]["legal_parameters"][-1] == parameter
        assert {item["source_id"] for item in payload["source_excerpts"]} == {"s_toy", "s_index"}
        raise ExpertRunError("SYNTHETIC inspected writer handoff")

    monkeypatch.setattr(runner, "_invoke", inspect_writer)
    with pytest.raises(ExpertRunError, match="SYNTHETIC inspected writer handoff"):
        await runner.create("A", ASSIGNMENT, [])

"""Synthetic native-tool source-gap reporting, without downloads or live models."""

import json

import pytest
from test_experts import completion, install_transport, tool_message
from test_experts import setup as setup

from taxcalcbench import experts
from taxcalcbench.experts import ExpertRunner, _SourceGap


@pytest.mark.parametrize("task", ["plan", "create", "review"])
async def test_source_gap_preserves_originating_stage_and_stops_native_run(setup, monkeypatch, task):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert len(requests) == 1
        return completion(tool_message("report_source_gap", {
            "reason": "SYNTHETIC missing historical index SYNTHETIC_TEST_KEY"}))

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    with pytest.raises(_SourceGap, match="SOURCE_GAP: SYNTHETIC missing historical index") as caught:
        await runner._invoke(task, "A", {}, experts._ProbeResult, runner._law_tools())
    assert caught.value.stage == task
    assert "SYNTHETIC_TEST_KEY" not in str(caught.value)
    assert runner.usage()["model_calls"] == 1
    record = json.loads(next(setup[3].glob("operations/*/error.json")).read_text())
    assert record["task"] == task and record["error"].startswith("SOURCE_GAP:")
    assert not list(setup[3].glob("operations/*/invalid_output.json"))


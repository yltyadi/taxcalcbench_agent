"""Native SDK retries preserve the current tool conversation and usage records."""

import json

import httpx
import pytest
from agents.run_internal import model_retry
from openai import AsyncOpenAI

from taxcalcbench import experts
from taxcalcbench.experts import ExpertRunError, ExpertRunner, ProviderUnavailable


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "SYNTHETIC_RETRY_SECRET")

    async def no_delay(_delay):
        pass

    monkeypatch.setattr(model_retry, "_sleep_for_retry", no_delay)
    return ExpertRunner({"runtime": {"model": "gemini-3.8-flash", "max_output_tokens": 1000}},
        {"documents": []}, tmp_path / "sources", tmp_path / "work", {})


def install_transport(monkeypatch, handler):
    def client(**kwargs):
        assert kwargs["max_retries"] == 0  # Only the Agents SDK retries; no nested client retry loop.
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(experts, "AsyncOpenAI", client)


def completion(message):
    return httpx.Response(200, json={"id": "synthetic-retry", "object": "chat.completion", "created": 1,
        "model": "gemini-3.8-flash", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})


async def test_retry_after_tool_response_keeps_history_and_meters_every_attempt(runner, monkeypatch):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return completion({"role": "assistant", "content": None, "tool_calls": [{
                "id": "synthetic-calculate", "type": "function",
                "function": {"name": "calculate", "arguments": '{"expression":"2+3"}'},
                "extra_content": {"google": {"thought_signature": "SYNTHETIC_SIGNATURE"}}}]})
        if len(requests) == 2:
            return httpx.Response(500, json={"error": {"message": "Internal error encountered.", "status": "INTERNAL"}})
        assert requests[1] == payload
        tool_result = next(message for message in payload["messages"] if message["role"] == "tool")
        assert tool_result["tool_call_id"] == "synthetic-calculate"
        assert "5" in tool_result["content"]
        original_call = next(message["tool_calls"][0] for message in payload["messages"] if message.get("tool_calls"))
        assert original_call["extra_content"]["google"]["thought_signature"] == "SYNTHETIC_SIGNATURE"
        return completion({"role": "assistant", "content": '{"value":"5"}'})

    install_transport(monkeypatch, handler)
    result = await runner.probe()
    assert result["status"] == "passed" and len(requests) == 3
    usage = json.loads((runner.work_dir / "usage.json").read_text())
    assert usage["model_calls"] == 3 and usage["reported_tokens"] == 36
    assert [call["status"] for call in usage["calls"]] == ["completed", "failed", "completed"]
    assert usage["accounted_tokens"] > usage["reported_tokens"]
    operation = next((runner.work_dir / "operations").iterdir())
    saved = json.loads((operation / "result.json").read_text())
    assert saved["calculation_calls"] == [{"expression": "2+3", "result": "5"}]
    assert len(list((operation / "provider_errors").glob("*.json"))) == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_persistent_temporary_failure_is_distinct_and_redacted(runner, monkeypatch, caplog, status):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "SYNTHETIC_RETRY_SECRET temporary failure"}})

    install_transport(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable, match=f"HTTP {status}") as caught:
        await runner.probe()
    assert len(requests) == 9 and all(payload == requests[0] for payload in requests)
    assert runner.usage()["model_calls"] == 9 and runner.usage()["reported_tokens"] == 0
    assert "SYNTHETIC_RETRY_SECRET" not in str(caught.value) + caplog.text
    for path in runner.work_dir.rglob("*.json"):
        assert "SYNTHETIC_RETRY_SECRET" not in path.read_text()


async def test_request_validation_failure_does_not_retry(runner, monkeypatch, caplog):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "SYNTHETIC_RETRY_SECRET invalid request"}})

    install_transport(monkeypatch, handler)
    with pytest.raises(ExpertRunError, match="HTTP 400") as caught:
        await runner.probe()
    assert not isinstance(caught.value, ProviderUnavailable)
    assert calls == 1 and runner.usage()["model_calls"] == 1
    assert "SYNTHETIC_RETRY_SECRET" not in str(caught.value) + caplog.text


async def test_connection_error_retries_then_pauses(runner, monkeypatch):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("SYNTHETIC connection refused", request=request)

    install_transport(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable, match="APIConnectionError"):
        await runner.probe()
    assert calls == 9 and runner.usage()["model_calls"] == 9


async def test_connection_recovers_after_old_retry_window_without_restarting(runner, monkeypatch, capsys):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) <= 5:
            raise httpx.ConnectError("SYNTHETIC temporary network failure", request=request)
        if len(requests) == 6:
            return completion({"role": "assistant", "content": None, "tool_calls": [{
                "id": "synthetic-recovery", "type": "function",
                "function": {"name": "calculate", "arguments": '{"expression":"2+3"}'}}]})
        return completion({"role": "assistant", "content": '{"value":"5"}'})

    install_transport(monkeypatch, handler)
    result = await runner.probe()
    assert result["status"] == "passed" and len(requests) == 7
    assert all(request == requests[0] for request in requests[:6])
    assert "retry 5/8" in capsys.readouterr().err


async def test_retry_count_is_configurable(runner, monkeypatch):
    runner.runtime["max_retries"] = 0
    install_transport(monkeypatch, lambda request: httpx.Response(503, json={"error": {"message": "Unavailable"}}))
    with pytest.raises(ProviderUnavailable):
        await runner.probe()
    assert runner.usage()["model_calls"] == 1


async def test_tool_call_id_collision_is_deduplicated_preventing_model_behavior_error(runner, monkeypatch):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return completion({"role": "assistant", "content": None, "tool_calls": [{
                "id": "reused-call-id", "type": "function",
                "function": {"name": "calculate", "arguments": '{"expression":"2+3"}'}}]})
        return completion({"role": "assistant", "content": '{"value":"5"}'})

    install_transport(monkeypatch, handler)
    # Simulate an earlier turn that had already used 'reused-call-id'
    original_model_init = experts._MeteredModel.__init__

    def metered_init(self, *args, **kwargs):
        original_model_init(self, *args, **kwargs)
        self.seen_call_ids.add("reused-call-id")

    monkeypatch.setattr(experts._MeteredModel, "__init__", metered_init)
    result = await runner.probe()
    assert result["status"] == "passed" and len(requests) == 2
    # The tool result must use the deduplicated ID, matching the assistant's rewritten tool call
    tool_result = next(message for message in requests[1]["messages"] if message["role"] == "tool")
    assert tool_result["tool_call_id"] != "reused-call-id"
    assert "reused-call-id_" in tool_result["tool_call_id"]

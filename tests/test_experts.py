"""Exercise native SDK tool/result turns against a local HTTP transport only."""

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from agents import AgentOutputSchema, function_tool
from agents.exceptions import ModelBehaviorError
from agents.run_internal import model_retry
from agents.tool_context import ToolContext
from openai import AsyncOpenAI, BadRequestError

from taxcalcbench import experts
from taxcalcbench.experts import BudgetExceeded, ExpertRunError, ExpertRunner, _Usage
from taxcalcbench.schema import UNITS, FamilyDraft, FamilyPlan
from taxcalcbench.tools import LawTools


def family_payload():
    return {"topic": "SYNTHETIC fixture only", "questions": [{
        "no": number, "family": "A-F01", "variant": "original" if number == 1 else "temporal" if number == 2 else "eligibility",
        "country": "XX", "language": "en", "tax_category": "Income, profits and capital gains",
        "primary_taxpayer": "Individual", "year_period_target": "2010-2019" if number == 2 else "2020-2026", "case_year": 2015 if number == 2 else 2025,
        "difficulty": "D3", "question": f"SYNTHETIC infrastructure fixture {1 if number == 2 else number}, period {{case_year}}, amount 2025 USD.",
        "final_answer_text": "SYNTHETIC result 5", "answer_value": "5", "unit": "USD",
        "gold_steps": [{"step": "SYNTHETIC 2+3", "result": "5", "unit": "USD", "citations": ["C1"]}],
        "status": "draft", "note": None if number == 1 else "SYNTHETIC condition change",
    } for number in range(1, 6)], "citations": [{
        "question_no": number, "language": "en", "citation_id": "C1", "official_title": "SYNTHETIC not law",
        "law_number": None, "article_section": "Synthetic section", "paragraph": None, "subparagraph": None,
        "rule_applies_from": "2025-01-01", "rule_applies_until": None,
        "official_source_url": None, "source_file": "s_toy", "supporting_passage": "SYNTHETIC fixture text.",
    } for number in range(1, 6)]}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    async def no_delay(_delay):
        pass

    monkeypatch.setattr(model_retry, "_sleep_for_retry", no_delay)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "SYNTHETIC_TEST_KEY")
    source = tmp_path / "source"
    source.mkdir()
    text = "Header\nSYNTHETIC fixture text.\nContinuation."
    (source / "raw.txt").write_text(text)
    (source / "text.txt").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    inventory = {"documents": [{"id": "s_toy", "title": "SYNTHETIC not law",
        "url": "https://synthetic.invalid/document", "final_url": "https://synthetic.invalid/document",
        "raw_path": "raw.txt", "text_path": "text.txt", "sha256": digest, "text_sha256": digest,
        "country": "XX", "language": "en", "kind": "law", "partial": True,
        "extraction_gaps": ["SYNTHETIC unavailable object"], "edition_note": "SYNTHETIC"}]}
    config = {"country": "XX", "language": "en", "generation": {
        "families": 1, "regular_questions_per_family": 5, "missing_information_per_family": 0,
        "categories": [{"code": "1000", "label": "Income, profits and capital gains", "weight": 1}],
        "periods": [{"label": "2020-2026", "start": 2020, "end": 2026, "weight": 4},
                    {"label": "2010-2019", "start": 2010, "end": 2019, "weight": 1}]}, "runtime": {
        "model": "gemini-3.8-flash", "max_model_calls": 80, "max_total_tokens": 1000000,
        "max_output_tokens": 16000, "timeout_seconds": 30, "max_turns": 8}}
    return config, inventory, source, tmp_path / "work"


def install_transport(monkeypatch, handler):
    def client(**kwargs):
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(experts, "AsyncOpenAI", client)


def completion(message, *, finish_reason=None):
    return httpx.Response(200, json={"id": "synthetic-completion", "object": "chat.completion",
        "created": 1, "model": "gemini-3.8-flash", "choices": [{"index": 0, "message": message,
            "finish_reason": finish_reason or ("tool_calls" if message.get("tool_calls") else "stop")}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})


def tool_message(name="calculate", arguments=None):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": "synthetic-call", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"expression": "2+3"})},
        "extra_content": {"google": {"thought_signature": "SYNTHETIC_OPAQUE_SIGNATURE"}}}]}


@pytest.mark.parametrize("effort", [None, "high"])
async def test_native_sdk_probe_preserves_provider_tool_metadata_and_uses_only_google_key(setup, monkeypatch, effort):
    requests = []
    if effort is not None:
        setup[0]["runtime"]["reasoning_effort"] = effort

    def handler(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.headers["authorization"] == "Bearer SYNTHETIC_TEST_KEY"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "gemini-3.8-flash"
        assert payload["tool_choice"] == ("auto" if len(requests) == 1 else "none")
        assert payload["reasoning_effort"] == (effort or "low")
        assert payload["response_format"]["json_schema"]["strict"] is True
        if len(requests) == 1:
            return completion(tool_message())
        call = next(message for message in payload["messages"] if message.get("tool_calls"))["tool_calls"][0]
        assert call["extra_content"]["google"]["thought_signature"] == "SYNTHETIC_OPAQUE_SIGNATURE"
        return completion({"role": "assistant", "content": '{"value":"5"}'})

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    result = await runner.probe()
    assert result["status"] == "passed"
    assert result["usage"]["model_calls"] == 2
    assert result["usage"]["reported_tokens"] == result["usage"]["accounted_tokens"] == 36
    saved = next(setup[3].glob("operations/*/result.json")).read_text()
    assert "SYNTHETIC_TEST_KEY" not in saved and "SYNTHETIC_OPAQUE_SIGNATURE" not in saved
    assert "2+3" in saved
    steps = [path.read_text() for path in setup[3].glob("operations/*/model_steps/*.json")]
    assert len(steps) == 2 and all("SYNTHETIC_OPAQUE_SIGNATURE" not in step for step in steps)
    assert all(json.loads(step)["reported_input_tokens"] == 11
        and json.loads(step)["reported_output_tokens"] == 7 for step in steps)
    metadata = [json.loads(path.read_text()) for path in setup[3].glob("operations/*/request_metadata/*.json")]
    assert len(metadata) == 2
    assert {item["tool_choice"] for item in metadata} == {"auto", "none"}
    assert all(item["response_format"] == "json_schema" for item in metadata)
    assert all(item["tools"] == ["calculate"] for item in metadata)
    assert all(item["reasoning_effort"] == (effort or "low") for item in metadata)


async def test_two_native_runs_share_atomic_usage_and_fresh_contexts(setup, monkeypatch):
    def handler(request):
        payload = json.loads(request.content)
        if not any(message["role"] == "tool" for message in payload["messages"]):
            return completion(tool_message())
        return completion({"role": "assistant", "content": '{"value":"5"}'})

    install_transport(monkeypatch, handler)
    config, inventory, source, work = setup
    limits = {"usage_file": str(work / "shared.json")}
    first = ExpertRunner(config, inventory, source, work / "a", limits)
    second = ExpertRunner(config, inventory, source, work / "b", limits)
    results = await asyncio.gather(first.probe(), second.probe())
    assert all(result["status"] == "passed" for result in results)
    assert first.usage()["model_calls"] == 4
    assert second.usage()["reported_tokens"] == 72


def test_shared_allowance_admission_is_atomic_and_failed_usage_is_retained(tmp_path):
    ledger = _Usage(tmp_path / "usage.json", 1, 100)

    def reserve(_):
        try:
            return ledger.reserve(10, 10, "SYNTHETIC")
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = [result for result in pool.map(reserve, range(2)) if result]
    assert len(accepted) == 1 and ledger.snapshot()["model_calls"] == 1
    ledger.finish(accepted[0], error="SYNTHETIC timeout")
    assert ledger.snapshot()["accounted_tokens"] == 20
    assert ledger.snapshot()["reported_tokens"] == 0
    with pytest.raises(BudgetExceeded):
        ledger.reserve(1, 1, "SYNTHETIC")


def test_usage_without_limits_records_completed_and_failed_calls(tmp_path):
    ledger = _Usage(tmp_path / "usage.json")
    first = ledger.reserve(1_000_000, 32_000, "SYNTHETIC")
    ledger.finish(first, tokens=80)
    second = ledger.reserve(2_000_000, 32_000, "SYNTHETIC")
    ledger.finish(second, error="SYNTHETIC failure")
    snapshot = ledger.snapshot()
    assert snapshot["max_model_calls"] is None and snapshot["max_total_tokens"] is None
    assert snapshot["model_calls"] == 2 and snapshot["reported_tokens"] == 80
    assert snapshot["accounted_tokens"] == 2_032_080


def test_usage_configuration_changes_keep_all_history_and_audit_caps(tmp_path):
    path = tmp_path / "usage.json"
    capped = _Usage(path, 1, 100)
    first = capped.reserve(10, 10, "SYNTHETIC")
    capped.finish(first, error="SYNTHETIC failure")
    before = json.loads(path.read_text())
    uncapped = _Usage(path)
    after = json.loads(path.read_text())
    assert after["calls"] == before["calls"]
    assert after["model_calls"] == before["model_calls"] == 1
    assert after["accounted_tokens"] == before["accounted_tokens"] == 20
    assert after["reported_tokens"] == before["reported_tokens"] == 0
    change = after["allowance_changes"][0]
    assert change["previous"] == {"max_model_calls": 1, "max_total_tokens": 100}
    assert change["current"] == {"max_model_calls": None, "max_total_tokens": None}
    assert "configuration change" in change["reason"] and change["changed_at"]
    # A previously constructed runner also follows the current shared configuration.
    second = capped.reserve(1000, 1000, "SYNTHETIC")
    uncapped.finish(second, tokens=50)
    limited_again = _Usage(path, 2, 100)
    assert len(limited_again.snapshot()["allowance_changes"]) == 2
    with pytest.raises(BudgetExceeded):
        uncapped.reserve(1, 1, "SYNTHETIC")
    assert len(json.loads(path.read_text())["calls"]) == 2


@pytest.mark.parametrize("max_calls,max_tokens", [(1, None), (None, 20)])
def test_each_usage_limit_can_be_enabled_independently(tmp_path, max_calls, max_tokens):
    ledger = _Usage(tmp_path / "usage.json", max_calls, max_tokens)
    ledger.reserve(10, 10, "SYNTHETIC")
    with pytest.raises(BudgetExceeded):
        ledger.reserve(1, 1, "SYNTHETIC")


def test_runner_without_budget_configuration_only_records_usage(setup):
    config = setup[0]
    config["runtime"].pop("max_model_calls")
    config["runtime"].pop("max_total_tokens")
    runner = ExpertRunner(*setup, {})
    assert runner.usage()["max_model_calls"] is None
    assert runner.usage()["max_total_tokens"] is None
    limited = ExpertRunner(*setup, {"max_model_calls": 3})
    assert limited.usage()["max_model_calls"] == 3
    assert limited.usage()["max_total_tokens"] is None


@pytest.mark.parametrize("max_calls,max_tokens", [(0, 100), (1, -1), (True, 100), (1.5, 100)])
def test_usage_rejects_invalid_limits(tmp_path, max_calls, max_tokens):
    with pytest.raises(ValueError, match="positive integers"):
        _Usage(tmp_path / "usage.json", max_calls, max_tokens)


def test_decoder_projection_preserves_exact_aliases_structure_enums_and_original_schema():
    original = AgentOutputSchema(FamilyDraft).json_schema()
    snapshot = json.dumps(original, sort_keys=True)
    projected = experts._DecoderSchema(FamilyDraft).json_schema()
    for model, schema in original["$defs"].items():
        assert set(projected["$defs"][model]["properties"]) == set(schema["properties"])
        assert projected["$defs"][model]["required"] == schema["required"]
        assert projected["$defs"][model]["additionalProperties"] is False
    assert projected["$defs"]["QuestionRow"]["properties"]["variant"]["enum"] == [
        "original", "numbers", "threshold", "eligibility", "temporal", "missing_information"]
    assert projected["properties"]["questions"]["items"] == original["properties"]["questions"]["items"]
    assert "maxLength" not in json.dumps(projected) and "pattern" not in json.dumps(projected)
    assert json.dumps(original, sort_keys=True) == snapshot
    # A property with the same name as a schema keyword is still a real field.
    assert experts._decoder_schema({"properties": {"minimum": {"type": "integer", "minimum": 1}}}) == {
        "properties": {"minimum": {"type": "integer"}}}


@pytest.mark.parametrize("field,value", [("case_year", 0), ("country", "invalid-country"),
                                        ("question", "x" * 40001), ("no", True)])
def test_projected_provider_schema_retains_strict_local_validation(field, value):
    payload = family_payload()
    payload["questions"][0][field] = value
    with pytest.raises(ModelBehaviorError):
        experts._DecoderSchema(FamilyDraft).validate_json(json.dumps(payload))


async def test_tool_decoder_defaults_are_omitted_without_changing_python_argument_defaults(setup):
    tool = function_tool(LawTools(setup[1], setup[2]).read_law)
    assert tool.params_json_schema["properties"]["end_line"]["default"] == 160
    tool.params_json_schema = experts._decoder_schema(tool.params_json_schema)
    assert "default" not in json.dumps(tool.params_json_schema)
    arguments = '{"source_id":"s_toy"}'
    context = ToolContext(context=None, tool_name="read_law", tool_call_id="synthetic", tool_arguments=arguments)
    result = await tool.on_invoke_tool(context, arguments)
    assert result["start_line"] == 1 and "Continuation." in result["text"]


async def test_provider_error_is_scrubbed_before_sdk_unwinds_and_reserved_usage_is_retained(setup, monkeypatch, caplog):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(400, json={"error": {"message": "SYNTHETIC_TEST_KEY private provider details", "type": "invalid_request_error"}})

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    with pytest.raises(ExpertRunError, match=r"BadRequestError \(HTTP 400\)") as caught:
        await runner.probe()
    assert count == 1
    assert "SYNTHETIC_TEST_KEY" not in str(caught.value) + caplog.text
    assert runner.usage()["model_calls"] == 1 and runner.usage()["reported_tokens"] == 0
    assert runner.usage()["accounted_tokens"] > 16000
    error = next(setup[3].glob("operations/*/error.json")).read_text()
    assert "private provider details" not in error
    diagnostic = json.loads(next(setup[3].glob("operations/*/provider_errors/*.json")).read_text())
    assert diagnostic["message"] == "[redacted] private provider details"
    assert diagnostic["category"] == "request_validation" and diagnostic["http_status"] == 400


def test_provider_diagnostics_keep_schema_reason_but_remove_every_configured_key_and_url_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "SYNTHETIC_GOOGLE_SECRET")
    monkeypatch.setenv("GEMINI_API_KEY", "SYNTHETIC_GEMINI_SECRET")
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC_OPENAI_SECRET")
    error = BadRequestError("Do not persist this raw exception", response=httpx.Response(400,
        request=httpx.Request("POST", "https://synthetic.invalid")), body={"error": {
            "message": "The specified schema produces a constraint that has too many states for serving. "
                "SYNTHETIC_GOOGLE_SECRET SYNTHETIC_GEMINI_SECRET SYNTHETIC_OPENAI_SECRET "
                "https://private-user:private-password@synthetic.invalid/docs?key=unlisted-secret#fragment",
            "code": "INVALID_ARGUMENT", "param": "response_format",
            "irrelevant_raw_headers": {"authorization": "Do not copy"}}})
    record = experts._provider_diagnostic(error)
    assert record["category"] == "schema_complexity"
    assert record["code"] == "INVALID_ARGUMENT" and record["param"] == "response_format"
    assert "too many states" in record["message"]
    assert "https://synthetic.invalid/docs" in record["message"]
    serialized = json.dumps(record)
    for secret in ("GOOGLE_SECRET", "GEMINI_SECRET", "OPENAI_SECRET", "private-user", "private-password",
                   "unlisted-secret", "irrelevant_raw_headers", "raw exception"):
        assert secret not in serialized


async def test_input_limit_stops_before_dispatch_or_reservation(setup, monkeypatch):
    setup[0]["runtime"]["max_input_chars"] = 1
    install_transport(monkeypatch, lambda request: pytest.fail("Provider was called"))
    runner = ExpertRunner(*setup, {})
    with pytest.raises(BudgetExceeded, match="input character"):
        await runner.probe()
    assert runner.usage()["model_calls"] == 0


def plan_payload():
    family = family_payload()
    return {"family_id": "A-F01", "topic": family["topic"], "cases": [{
        "question_no": row["no"], "variant": row["variant"], "case_year": row["case_year"],
        "tax_category": row["tax_category"], "primary_taxpayer": row["primary_taxpayer"],
        "scenario_outline": f"SYNTHETIC scenario {row['no']}",
        "legal_difference": f"SYNTHETIC condition {row['no']}",
        "legal_assumptions": ["SYNTHETIC fixture assumption"],
        "derived_from": 1 if row["variant"] == "temporal" else None,
        "evidence": [{"source_id": "s_toy", "start_line": 1, "end_line": 3,
                      "purpose": "SYNTHETIC saved evidence"}],
        "legal_parameters": [{"name": "SYNTHETIC statutory rate", "value": "5%",
            "evidence": [{"source_id": "s_toy", "start_line": 1, "end_line": 3,
                          "purpose": "SYNTHETIC parameter evidence"}]}],
    } for row in family["questions"]]}


def repeat_invalid_plan(monkeypatch, runner, plan):
    """One local repair may repeat a bad plan, but must never reach a writer."""
    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        assert task == "plan" and payload["previous_plan"] and payload["feedback"]
        return FamilyPlan.model_validate(plan)

    monkeypatch.setattr(runner, "_invoke", invoke)


async def test_invalid_native_plan_repairs_once_with_full_research_and_provider_metadata(setup, monkeypatch):
    requests = []
    invalid = plan_payload()
    invalid["cases"][1]["derived_from"] = None

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return completion(tool_message("read_law", {"source_id": "s_toy", "start_line": 1, "end_line": 3}))
        if len(requests) == 2:
            return completion({"role": "assistant", "content": json.dumps(invalid)})
        assert len(requests) == 3
        assert not body.get("tools") and body["tool_choice"] == "none"
        assert body["response_format"]["json_schema"]["strict"] is True
        call = next(item["tool_calls"][0] for item in body["messages"] if item.get("tool_calls"))
        assert call["extra_content"]["google"]["thought_signature"] == "SYNTHETIC_OPAQUE_SIGNATURE"
        assert any(item["role"] == "tool" and "SYNTHETIC fixture text" in item["content"] for item in body["messages"])
        assert any(item["role"] == "assistant" and item.get("content") == json.dumps(invalid) for item in body["messages"])
        assert "cases[1]" in body["messages"][-1]["content"]
        assert "arbitrary derived_from" in body["messages"][-1]["content"]
        return completion({"role": "assistant", "content": json.dumps(plan_payload())})

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    tools = runner._law_tools()
    result = await runner._invoke("plan", "A", {"assignment": ASSIGNMENT}, FamilyPlan, tools)
    assert result.cases[1].derived_from == 1
    assert runner.usage()["model_calls"] == 3
    saved = json.loads(next(setup[3].glob("operations/*/result.json")).read_text())
    assert saved["read_source_ids"] == ["s_toy"]
    assert not list(setup[3].glob("operations/*/error.json"))
    invalid_path = next(setup[3].glob("operations/*/invalid_output.json"))
    invalid_record = json.loads(invalid_path.read_text())
    assert invalid_record["attempt"] == 1 and json.loads(invalid_record["output_text"]) == invalid
    assert invalid_record["validation_errors"][0]["path"] == "cases[1]"
    assert "require derived_from" in invalid_record["validation_errors"][0]["message"]
    assert "SYNTHETIC_OPAQUE_SIGNATURE" not in "".join(path.read_text() for path in invalid_path.parent.rglob("*.json"))


async def test_repeated_invalid_output_stops_after_one_repair_and_reports_safe_field_errors(setup, monkeypatch):
    requests = []
    invalid = plan_payload()
    invalid["cases"][1]["derived_from"] = None
    invalid["cases"][0]["case_year"] = "PRIVATE_REJECTED_VALUE"
    invalid["PRIVATE_UNTRUSTED_FIELD_NAME"] = "SYNTHETIC_TEST_KEY"

    def handler(request):
        requests.append(json.loads(request.content))
        assert len(requests) <= 2
        return completion({"role": "assistant", "content": json.dumps(invalid)})

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    with pytest.raises(ExpertRunError, match=r"cases\[1\].*require derived_from") as caught:
        await runner._invoke("plan", "A", {}, FamilyPlan, runner._law_tools())
    assert len(requests) == runner.usage()["model_calls"] == 2
    error_path = next(setup[3].glob("operations/*/error.json"))
    error_text = error_path.read_text() + str(caught.value)
    assert "<extra_field>" in error_text
    for secret in ("SYNTHETIC_TEST_KEY", "PRIVATE_REJECTED_VALUE", "PRIVATE_UNTRUSTED_FIELD_NAME"):
        assert secret not in error_text
    invalid_record = json.loads((error_path.parent / "invalid_output.json").read_text())
    assert invalid_record["attempt"] == 2
    assert json.loads(invalid_record["output_text"])["PRIVATE_UNTRUSTED_FIELD_NAME"] == "[redacted]"


async def test_repair_budget_failure_keeps_budget_type_and_saved_invalid_draft(setup, monkeypatch):
    invalid = plan_payload()
    invalid["cases"][1]["derived_from"] = None
    install_transport(monkeypatch, lambda request: completion({"role": "assistant", "content": json.dumps(invalid)}))
    runner = ExpertRunner(*setup, {"max_model_calls": 1})
    with pytest.raises(BudgetExceeded):
        await runner._invoke("plan", "A", {}, FamilyPlan, runner._law_tools())
    assert runner.usage()["model_calls"] == 1
    record = json.loads(next(setup[3].glob("operations/*/error.json")).read_text())
    assert record["validation_errors"][0]["path"] == "cases[1]"
    assert list(setup[3].glob("operations/*/invalid_output.json"))


@pytest.mark.parametrize("valid", [True, False])
async def test_saved_draft_repair_only_never_restarts_research_or_recurses(setup, monkeypatch, valid):
    requests = []
    output = plan_payload()
    if not valid:
        output["cases"][1]["derived_from"] = None

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert len(requests) == 1
        assert not body.get("tools") and body["tool_choice"] == "none"
        assert "saved draft" in body["messages"][0]["content"]
        assert "previous_output" in body["messages"][1]["content"]
        return completion({"role": "assistant", "content": json.dumps(output)})

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    call = runner._invoke("plan", "A", {"previous_output": output}, FamilyPlan,
                          runner._law_tools(), repair_only=True)
    if valid:
        assert (await call).cases[1].derived_from == 1
    else:
        with pytest.raises(ExpertRunError, match="require derived_from"):
            await call
    assert runner.usage()["model_calls"] == 1


def question_payload(number):
    family = family_payload()
    return {"question": family["questions"][number - 1],
            "citations": [family["citations"][number - 1]]}


def review_payload():
    return {"family_id": "A-F01", "reviewer": "B", "family_issues": [], "questions": [
        {"question_no": number, "passed": True, "issues": [], "recomputed_answer": "5.00",
         "unit": "USD", "calculation": "1+4"} for number in range(1, 6)]}


def install_workflow_transport(monkeypatch, setup, requests, *, truncate_question=None):
    """Native SDK turns with synthetic outputs and actual fixture-file evidence."""
    saved_text = (setup[2] / "text.txt").read_text()

    def handler(request):
        body = json.loads(request.content)
        user = json.loads(next(item["content"] for item in body["messages"] if item["role"] == "user"))
        stage = "review" if "reviewer" in user else "create" if "case" in user else "plan"
        number = user["case"]["question_no"] if stage == "create" else None
        requests.append((stage, number, body))
        assert body.get("max_tokens", body.get("max_completion_tokens")) == setup[0]["runtime"]["max_output_tokens"]
        if body["tool_choice"] == "none":
            assert not body.get("tools")
        else:
            expected_tools = {"read_sections", "read_law", "find_in_laws", "list_sections"}
            if stage != "plan":
                expected_tools.add("calculate_many")
            assert expected_tools.issubset(item["function"]["name"] for item in body["tools"])
        assert user["sources"][0]["partial"] is True
        if stage != "plan":
            assert user["source_excerpts"][0]["text"] == saved_text
            assert "previous_topics" not in user
        if stage == "create":
            original = user["original_question"]
            assert original is None if number == 1 else original["no"] == 1
            assert user["required_question_template"] == (question_payload(1)["question"]["question"] if number == 2 else None)
            assert "family" not in user  # Each writer receives only its assigned case plus the Original.
        if stage == "review":
            assert user["family_plan"]["family_id"] == "A-F01"
            assert "source_review" not in user
            assert len(user["lifecycle_excerpts"]) == 1
        if not any(item["role"] == "tool" for item in body["messages"]):
            assert [item["role"] for item in body["messages"]] == ["system", "user"]
            if stage == "plan":
                return completion(tool_message("read_law", {"source_id": "s_toy", "start_line": 1, "end_line": 3}))
            if stage == "review":
                # Fresh arithmetic differs from the writers' calculation receipts.
                return completion(tool_message("calculate_many", {"expressions": ["1+4"] * 5}))
            return completion(tool_message("calculate", {"expression": "2+3"}))
        if stage == "create" and number == truncate_question:
            return completion({"role": "assistant", "content": '{"question":'}, finish_reason="length")
        output = (review_payload() if stage == "review"
                  else question_payload(number) if stage == "create" else plan_payload())
        return completion({"role": "assistant", "content": json.dumps(output)})

    install_transport(monkeypatch, handler)


ASSIGNMENT = {"family_id": "A-F01", "question_numbers": [1, 2, 3, 4, 5],
              "regular_question_numbers": [1, 2, 3, 4, 5], "missing_question_numbers": [],
              "period_slots": [{"question_no": no, "label": "2010-2019" if no == 2 else "2020-2026",
                                "start": 2010 if no == 2 else 2020, "end": 2019 if no == 2 else 2026}
                               for no in range(1, 6)]}


@pytest.mark.parametrize("review_effort", [None, "high"])
async def test_native_plan_five_writers_and_review_use_fresh_contexts_and_saved_evidence(setup, monkeypatch, review_effort):
    requests = []
    setup[0]["runtime"]["max_output_tokens"] = 32000
    if review_effort is not None:
        setup[0]["runtime"]["review_reasoning_effort"] = review_effort
    install_workflow_transport(monkeypatch, setup, requests)
    runner = ExpertRunner(*setup, {})
    family = await runner.create("A", ASSIGNMENT, [])
    assert isinstance(family, FamilyDraft) and len(family.questions) == 5
    assert "period 2025, amount 2025 USD" in family.questions[0].question
    assert "period 2015, amount 2025 USD" in family.questions[1].question
    assert all("{case_year}" not in question.question for question in family.questions)
    saved = [json.loads((setup[3] / "questions" / "A-F01" / f"{number}.json").read_text()) for number in (1, 2)]
    assert saved[0]["question_template"] == saved[1]["question_template"]
    assert "{case_year}" in saved[0]["question_template"]
    review = await runner.review("B", family, [{"family_id": "B-F01", "topic": "SYNTHETIC other"}])
    assert not review.family_issues and all(question.passed for question in review.questions)
    assert runner.usage()["model_calls"] == 14
    assert [(task, number) for task, number, _ in requests[::2]] == [
        ("plan", None), *[("create", number) for number in range(1, 6)],
        ("review", None)]
    # A reviewer-specific setting must reach every native SDK request without
    # changing the planner or writers, which retain the default effort.
    assert all(body["reasoning_effort"] == (review_effort or "low" if task == "review" else "low")
        for task, _, body in requests)
    for path in setup[3].glob("operations/*/request_metadata/*.json"):
        expected = (review_effort or "low" if path.parent.parent.name.startswith("review-") else "low")
        assert json.loads(path.read_text())["reasoning_effort"] == expected
    records = [json.loads(path.read_text()) for path in setup[3].glob("operations/*/result.json")]
    assert next(record for record in records if record["task"] == "plan")["read_source_ids"] == ["s_toy"]
    writers = [record for record in records if record["task"] == "create"]
    assert len(writers) == 5
    assert all(record["supplied_source_ids"] == ["s_toy"] and record["read_source_ids"] == [] for record in writers)
    assert all(record["calculation_calls"] == [{"expression": "2+3", "result": "5"}] for record in writers)
    reviewer = next(record for record in records if record["task"] == "review")
    assert reviewer["supplied_source_ids"] == ["s_toy"]
    assert reviewer["calculation_calls"] == [{"expression": "1+4", "result": "5"}] * 5


async def test_truncated_question_resume_reuses_plan_and_completed_rows_then_reviews(setup, monkeypatch):
    first_requests = []
    install_workflow_transport(monkeypatch, setup, first_requests, truncate_question=3)
    runner = ExpertRunner(*setup, {})
    with pytest.raises(ExpertRunError, match="ModelBehaviorError"):
        await runner.create("A", ASSIGNMENT, [])
    plan_path = setup[3] / "plans" / "A-F01.json"
    rows = setup[3] / "questions" / "A-F01"
    assert plan_path.exists() and {path.name for path in rows.iterdir()} == {"1.json", "2.json"}
    saved = {path.name: path.read_bytes() for path in rows.iterdir()}
    before_calls = runner.usage()["model_calls"]
    assert before_calls == 9  # Completed research plus one format-only correction.
    assert not any(task == "review" for task, _, _ in first_requests)
    failed = next(path for path in setup[3].glob("operations/*/error.json"))
    assert json.loads(failed.read_text())["calculation_calls"] == [{"expression": "2+3", "result": "5"}]
    assert any(item.get("text") == '{"question":'
        for path in failed.parent.glob("model_steps/*.json") for item in json.loads(path.read_text())["items"])

    resumed_requests = []
    install_workflow_transport(monkeypatch, setup, resumed_requests)
    resumed = ExpertRunner(*setup, {})
    family = await resumed.create("A", ASSIGNMENT, [])
    report = await resumed.review("B", family, [])
    assert not report.family_issues and all(item.passed for item in report.questions)
    assert [(task, number) for task, number, _ in resumed_requests[::2]] == [
        ("create", 3), ("create", 4), ("create", 5),
        ("review", None)]
    assert all((rows / name).read_bytes() == contents for name, contents in saved.items())
    assert resumed.usage()["model_calls"] == before_calls + 8
    assert len([path for path in rows.glob("*.json") if path.stem.isdecimal()]) == 5
    assert not list(rows.glob("*.source-review.json"))
    reused = await resumed.create("A", ASSIGNMENT, [])
    assert reused == family and resumed.usage()["model_calls"] == before_calls + 8


async def test_completed_writer_response_recovers_after_question_checkpoint_write_fails(setup, monkeypatch):
    first_requests = []
    install_workflow_transport(monkeypatch, setup, first_requests)
    runner = ExpertRunner(*setup, {})
    rows = setup[3] / "questions/A-F01"
    write = experts._write
    interrupted = False

    def interrupt_checkpoint(path, value):
        nonlocal interrupted
        if path == rows / "2.json" and not interrupted:
            interrupted = True
            raise OSError("SYNTHETIC interruption before question checkpoint")
        write(path, value)

    monkeypatch.setattr(experts, "_write", interrupt_checkpoint)
    with pytest.raises(OSError, match="before question checkpoint"):
        await runner.create("A", ASSIGNMENT, [])
    assert {path.name for path in rows.iterdir()} == {"1.json"}
    original = (rows / "1.json").read_bytes()
    operation = next(path.parent for path in setup[3].glob("operations/create-A-*/result.json")
                     if json.loads(path.read_text())["output"]["question"]["no"] == 2)
    saved_result = json.loads((operation / "result.json").read_text())
    assert saved_result["status"] == "completed"
    assert saved_result["calculation_calls"] == [{"expression": "2+3", "result": "5"}]
    before_calls = runner.usage()["model_calls"]
    assert before_calls == 6

    resumed_requests = []
    install_workflow_transport(monkeypatch, setup, resumed_requests)
    resumed = ExpertRunner(*setup, {})
    # A prompt fix must not discard a completed response for identical work.
    prompts = setup[3].parent / "updated-prompts"
    prompts.mkdir()
    (prompts / "create.md").write_text((resumed.prompt_dir / "create.md").read_text()
                                       + "\nSYNTHETIC revised wording guidance.\n")
    resumed.prompt_dir = prompts
    family = await resumed.create("A", ASSIGNMENT, [])
    assert [(task, number) for task, number, _ in resumed_requests[::2]] == [
        ("create", 3), ("create", 4), ("create", 5)]
    assert resumed.usage()["model_calls"] == before_calls + 6
    assert (rows / "1.json").read_bytes() == original
    recovered = json.loads((rows / "2.json").read_text())
    expected = saved_result["output"]
    expected["question"]["question"] = expected["question"]["question"].replace("{case_year}", "2015")
    assert recovered["output"] == expected
    assert len(family.questions) == 5
    assert await resumed.create("A", ASSIGNMENT, []) == family
    assert resumed.usage()["model_calls"] == before_calls + 6

    # The recovered response is unsuitable when the shared scenario changes.
    payload = json.loads((operation / "input.json").read_text())["input"]
    assert resumed._saved_output("create", "A", payload) is not None
    assert resumed._saved_output("create", "A", payload | {
        "required_question_template": payload["required_question_template"] + " Different synthetic facts."
    }) is None
    assert resumed._saved_output("create", "A", payload | {
        "case": payload["case"] | {"case_year": 2016}
    }) is None


async def test_saved_plan_rechecks_current_evidence_before_writing(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    experts._write(setup[3] / "plans" / "A-F01.json", plan_payload())
    (setup[2] / "text.txt").write_text("Changed source")
    install_transport(monkeypatch, lambda request: pytest.fail("Model must not run with modified evidence"))
    with pytest.raises(ExpertRunError, match="checksum"):
        await runner.create("A", ASSIGNMENT, [])
    assert runner.usage()["model_calls"] == 0


@pytest.mark.parametrize("mismatch", ["country", "language", "navigation", "missing_span", "extraction_gap"])
async def test_saved_plan_rejects_unusable_evidence_before_writing(setup, monkeypatch, mismatch):
    plan = plan_payload()
    document = setup[1]["documents"][0]
    if mismatch in ("country", "language"):
        document[mismatch] = "different"
    elif mismatch == "navigation":
        document["kind"] = "navigation"
    elif mismatch == "missing_span":
        plan["cases"][0]["evidence"][0]["end_line"] = 4
    else:
        document["extraction_gaps"] = [{"start": 8, "end": 20, "label": "SYNTHETIC missing provision"}]
    experts._write(setup[3] / "plans" / "A-F01.json", plan)
    runner = ExpertRunner(*setup, {})
    repeat_invalid_plan(monkeypatch, runner, plan)
    with pytest.raises(ExpertRunError, match="official law|missing or unreadable"):
        await runner.create("A", ASSIGNMENT, [])
    assert runner.usage()["model_calls"] == 0


async def test_writer_draft_without_calculator_receipt_is_saved_for_independent_review(setup, monkeypatch):
    setup[0]["generation"]["min_periods_per_family"] = 1
    plan = plan_payload()
    plan["cases"] = plan["cases"][:1]
    experts._write(setup[3] / "plans/A-F01.json", plan)
    assignment = ASSIGNMENT | {"question_numbers": [1], "regular_question_numbers": [1],
                               "period_slots": ASSIGNMENT["period_slots"][:1]}
    install_transport(monkeypatch, lambda request: completion({
        "role": "assistant", "content": json.dumps(question_payload(1))}))
    runner = ExpertRunner(*setup, {})
    family = await runner.create("A", assignment, [])
    assert family.questions[0].status == "draft"
    assert runner.usage()["model_calls"] == 1
    assert (setup[3] / "questions/A-F01/1.json").exists()


@pytest.mark.parametrize("currency", ["USD", "SYNTHETIC_NEW_CURRENCY"])
async def test_writer_receives_configured_currency_and_exact_accepted_units(setup, monkeypatch, currency):
    setup[0]["currency"] = currency
    experts._write(setup[3] / "plans/A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})

    async def inspect_writer(task, expert_id, payload, output_type, tools, **kwargs):
        assert task == "create"
        assert payload["currency"] == currency
        assert payload["allowed_units"] == sorted(set(UNITS) | {currency})
        raise ExpertRunError("SYNTHETIC inspected currency handoff")

    monkeypatch.setattr(runner, "_invoke", inspect_writer)
    with pytest.raises(ExpertRunError, match="SYNTHETIC inspected currency handoff"):
        await runner.create("A", ASSIGNMENT, [])


async def test_writer_changing_planned_case_cannot_poison_resume_checkpoint(setup, monkeypatch):
    experts._write(setup[3] / "plans" / "A-F01.json", plan_payload())

    def handler(request):
        body = json.loads(request.content)
        if not any(item["role"] == "tool" for item in body["messages"]):
            return completion(tool_message())
        output = question_payload(1)
        output["question"]["case_year"] = 2024
        return completion({"role": "assistant", "content": json.dumps(output)})

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    with pytest.raises(ExpertRunError, match="changed the assigned plan identity"):
        await runner.create("A", ASSIGNMENT, [])
    assert not (setup[3] / "questions" / "A-F01" / "1.json").exists()


async def test_writer_returning_next_case_gets_targeted_repair_before_checkpoint(setup, monkeypatch):
    experts._write(setup[3] / "plans" / "A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(payload)
        number = payload["case"]["question_no"]
        output = question_payload(number)
        if len(calls) == 1:
            output["question"]["no"] = 2
            output["question"]["case_year"] = 2016
            for citation in output["citations"]:
                citation["question_no"] = 2
        elif len(calls) == 2:
            assert number == 1
            assert any("Required identity" in item for item in payload["feedback"])
            assert payload["previous_question"]["no"] == 2
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    draft = await runner.create("A", ASSIGNMENT, [])
    assert len(calls) == 6  # five assigned cases and one focused correction
    assert [row.no for row in draft.questions] == [1, 2, 3, 4, 5]
    saved = json.loads((setup[3] / "questions/A-F01/1.json").read_text())
    assert saved["output"]["question"]["no"] == 1
    assert saved["output"]["question"]["case_year"] == plan_payload()["cases"][0]["case_year"]


async def test_completed_identity_repair_survives_interruption_before_question_checkpoint(setup, monkeypatch):
    experts._write(setup[3] / "plans/A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})
    calls = []
    original_write = experts._write

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(payload["case"]["question_no"])
        output = question_payload(payload["case"]["question_no"])
        if len(calls) == 1:
            output["question"]["case_year"] = 2016
        directory = setup[3] / "operations" / f"create-A-synthetic{len(calls)}"
        original_write(directory / "input.json", {"model": runner.model, "input": payload})
        original_write(directory / "result.json", {"output": output})
        return output_type.model_validate(output)

    def interrupted_write(path, value):
        if path == setup[3] / "questions/A-F01/1.json":
            raise KeyboardInterrupt
        original_write(path, value)

    monkeypatch.setattr(runner, "_invoke", invoke)
    monkeypatch.setattr(experts, "_write", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        await runner.create("A", ASSIGNMENT, [])
    assert calls == [1, 1]
    monkeypatch.setattr(experts, "_write", original_write)
    draft = await runner.create("A", ASSIGNMENT, [])
    assert calls == [1, 1, 2, 3, 4, 5]  # no charge to repeat the finished correction
    assert draft.questions[0].case_year == 2025


async def test_planning_control_flag_does_not_rewrite_unchanged_questions(setup, monkeypatch):
    experts._write(setup[3] / "plans/A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        assert task == "create"
        number = payload["case"]["question_no"]
        calls.append(number)
        return output_type.model_validate(question_payload(number))

    monkeypatch.setattr(runner, "_invoke", invoke)
    first = await runner.create("A", ASSIGNMENT, [])
    calls.clear()
    await runner.create("A", ASSIGNMENT, [], previous=first,
                        feedback=["Reviewer requests a revised family plan", "Question 5: Synthetic correction"])
    assert calls == [5]


@pytest.mark.parametrize("expression", ["2+3", "2+4"])
async def test_one_review_replays_arithmetic_without_a_tool_receipt(setup, monkeypatch, expression):
    from taxcalcbench.pipeline import review_failures

    output = review_payload()
    for question in output["questions"]:
        question["calculation"] = expression
    requests = []

    def handler(request):
        requests.append(request)
        return completion({"role": "assistant", "content": json.dumps(output)})

    install_transport(monkeypatch, handler)
    family = FamilyDraft.model_validate(family_payload())
    report = await ExpertRunner(*setup, {}).review("B", family, [])
    errors = review_failures(family, report, "A")
    assert len(requests) == 1
    assert bool(errors) == (expression == "2+4")
    assert all("calculation and answer disagree" in error for error in errors)


async def test_source_gap_tool_propagates_specific_reason_without_extra_model_turn(setup, monkeypatch):
    install_transport(monkeypatch, lambda request: completion(tool_message("report_source_gap", {"reason": "SYNTHETIC commencement source unavailable"})))
    runner = ExpertRunner(*setup, {})
    with pytest.raises(ExpertRunError, match="SOURCE_GAP: SYNTHETIC commencement source unavailable"):
        await runner.create("A", ASSIGNMENT, [])
    assert runner.usage()["model_calls"] == 1


async def test_exhausted_probe_saves_visible_steps_and_tool_receipts_without_reasoning(setup, monkeypatch):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        message = tool_message()
        message["tool_calls"][0]["id"] = f"synthetic-{count}"
        message["reasoning_content"] = "SYNTHETIC private reasoning must not be persisted"
        return completion(message)

    install_transport(monkeypatch, handler)
    runner = ExpertRunner(*setup, {})
    with pytest.raises(ExpertRunError, match="MaxTurnsExceeded"):
        await runner.probe()
    assert runner.usage()["model_calls"] == 5
    operation = next(setup[3].glob("operations/*"))
    error = json.loads((operation / "error.json").read_text())
    assert len(error["calculation_calls"]) == 5
    steps = sorted((operation / "model_steps").glob("*.json"))
    assert len(steps) == 5
    assert all(json.loads(path.read_text())["items"][0]["name"] == "calculate" for path in steps)
    persisted = "".join(path.read_text() for path in operation.rglob("*.json"))
    assert "SYNTHETIC private reasoning" not in persisted
    assert "SYNTHETIC_OPAQUE_SIGNATURE" not in persisted


@pytest.mark.parametrize("problem", ["period", "no_contrast", "counts"])
async def test_plans_cannot_skip_assigned_temporal_or_missing_constraints(setup, monkeypatch, problem):
    from copy import deepcopy

    payload = plan_payload()
    assignment = deepcopy(ASSIGNMENT)
    if problem == "period":
        payload["cases"][1]["case_year"] = 2024
    elif problem == "no_contrast":
        payload["cases"][1].update(variant="eligibility", derived_from=None)
    else:
        assignment["missing_question_numbers"] = [5]
        assignment["regular_question_numbers"] = [1, 2, 3, 4]
    experts._write(setup[3] / "plans/A-F01.json", payload)
    runner = ExpertRunner(*setup, {})
    repeat_invalid_plan(monkeypatch, runner, payload)
    with pytest.raises(ExpertRunError, match="assigned period|temporal counterpart|counts"):
        await runner.create("A", assignment, [])


@pytest.mark.parametrize("reason,enabled,passes", [(None, True, False), ("SYNTHETIC evidence supports this category.", True, True),
                                                  ("SYNTHETIC rationale.", False, False)])
async def test_planner_category_reallocation_is_explicit_enabled_and_recorded(setup, monkeypatch, reason, enabled, passes):
    payload = plan_payload()
    target = "Property taxes"
    setup[0]["generation"]["categories"].append({"code": "4000", "label": target, "enabled": enabled})
    for case in payload["cases"]:
        case["tax_category"] = target
    payload["category_reallocation_reason"] = reason
    experts._write(setup[3] / "plans/A-F01.json", payload)
    assignment = ASSIGNMENT | {"category": "Income, profits and capital gains"}
    runner = ExpertRunner(*setup, {})
    repeat_invalid_plan(monkeypatch, runner, payload)
    if passes:
        assert (await runner.plan("A", assignment, [])).cases[0].tax_category == target
        records = json.loads((setup[3] / "category_reallocations.json").read_text())
        assert records["A-F01"] == {"assigned_category": assignment["category"], "selected_category": target, "reason": reason}
    else:
        with pytest.raises(ExpertRunError, match="category|enabled"):
            await runner.plan("A", assignment, [])


@pytest.mark.parametrize("repair", [False, True])
async def test_source_gap_retry_does_not_treat_failed_topic_as_a_category_requirement(setup, monkeypatch, repair):
    target = "Property taxes"
    setup[0]["generation"]["categories"].append({"code": "4000", "label": target})
    assignment = ASSIGNMENT | {"category": "Income, profits and capital gains",
        "category_options": ["Income, profits and capital gains", target]}
    gap = "SOURCE_GAP: SYNTHETIC missing rate. Changing category needs permission."
    runner = ExpertRunner(*setup, {})
    calls = 0

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        nonlocal calls
        calls += 1
        assert "category" not in payload["assignment"]
        assert payload["assignment"]["category_options"] == assignment["category_options"]
        assert payload["feedback"][0] == "Keep the assigned years."
        assert gap not in payload["feedback"]
        assert payload["previous_source_gaps"] == [gap]
        assert "already have permission" in payload["replanning_instruction"]
        result = plan_payload()
        for case in result["cases"]:
            case["tax_category"] = target
        result["category_reallocation_reason"] = "SYNTHETIC alternative has readable evidence."
        if repair and calls == 1:
            result["category_reallocation_reason"] = None
        return output_type.model_validate(result)

    monkeypatch.setattr(runner, "_invoke", invoke)
    plan = await runner.plan("A", assignment, [], feedback=[gap, "Keep the assigned years."])
    assert calls == 1 + int(repair)
    assert plan.cases[0].tax_category == target
    assert assignment["category"] == "Income, profits and capital gains"
    records = json.loads((setup[3] / "category_reallocations.json").read_text())
    assert records["A-F01"]["assigned_category"] == assignment["category"]


@pytest.mark.parametrize("problem", ["paraphrase", "amount", "placeholder", "wrong_year"])
async def test_writer_recovers_rendered_template_but_saves_real_conflicts_for_correction(setup, monkeypatch, problem):
    experts._write(setup[3] / "plans/A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        number = payload["case"]["question_no"]
        output = question_payload(number)
        if number == 2:
            if problem == "paraphrase":
                output["question"]["question"] += " Rephrased."
            elif problem == "amount":
                output["question"]["question"] = output["question"]["question"].replace("amount 2025", "amount 2015")
            elif problem == "placeholder":
                output["question"]["question"] = output["question"]["question"].replace("{case_year}", "2015")
            else:
                output["question"]["question"] = output["question"]["question"].replace("{case_year}", "2014")
        tools.calculate("2+3")
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    draft = await runner.create("A", ASSIGNMENT, [])
    assert all(row.status == "draft" for row in draft.questions)
    parent = json.loads((setup[3] / "questions/A-F01/1.json").read_text())
    child = json.loads((setup[3] / "questions/A-F01/2.json").read_text())
    if problem == "placeholder":
        assert parent["question_template"] == child["question_template"]
        assert draft.questions[1].question == parent["question_template"].replace("{case_year}", "2015")
    else:
        assert parent["question_template"] != child["question_template"]
    if problem == "wrong_year":
        assert "period 2014" in draft.questions[1].question
    assert (setup[3] / "questions/A-F01/1.json").exists()


@pytest.mark.parametrize("number", [1, 2, 3])
async def test_completed_rendered_correction_recovers_template_without_changing_year_valued_facts(setup, monkeypatch, number):
    experts._write(setup[3] / "plans/A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})
    year = 2015 if number == 2 else 2025

    async def initial(task, expert, payload, output_type, tools, **kwargs):
        output = question_payload(payload["case"]["question_no"])
        output["question"]["question"] = output["question"]["question"].replace(
            "amount 2025", f"amount {year}") + f" Fixed reference date {year}."
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", initial)
    previous = await runner.create("A", ASSIGNMENT, [])
    paths = setup[3] / "questions/A-F01"
    before = {path.name: path.read_bytes() for path in paths.glob("*.json")}
    template = json.loads(before[f"{number}.json"])["question_template"]
    feedback = [f"Question {number}: correct the synthetic citation locator only"]

    async def corrected_then_interrupted(task, expert, payload, output_type, tools, **kwargs):
        assert payload["case"]["question_no"] == number
        output = {
            "question": next(row.model_dump(mode="json") for row in previous.questions if row.no == number),
            "citations": [row.model_dump(mode="json") for row in previous.citations if row.question_no == number],
        }
        output["citations"][0]["article_section"] = "Corrected SYNTHETIC locator"
        operation = setup[3] / "operations/create-A-synthetic-completed-correction"
        experts._write(operation / "input.json", {"model": runner.model, "input": payload})
        experts._write(operation / "result.json", {"output": output})
        raise ExpertRunError("SYNTHETIC interruption after completed correction")

    monkeypatch.setattr(runner, "_invoke", corrected_then_interrupted)
    with pytest.raises(ExpertRunError, match="SYNTHETIC interruption"):
        await runner.create("A", ASSIGNMENT, [], feedback=feedback, previous=previous)

    async def no_model(*args, **kwargs):
        pytest.fail("The completed correction must be recovered without another model call")

    monkeypatch.setattr(runner, "_invoke", no_model)
    recovered = await runner.create("A", ASSIGNMENT, [], feedback=feedback, previous=previous)
    assert recovered.questions == previous.questions
    assert next(c for c in recovered.citations if c.question_no == number).article_section == "Corrected SYNTHETIC locator"
    saved = json.loads((paths / f"{number}.json").read_text())
    assert saved["question_template"] == template
    assert f"amount {year} USD" in template and f"Fixed reference date {year}" in template
    assert all((paths / name).read_bytes() == data for name, data in before.items() if name != f"{number}.json")
    assert await runner.create("A", ASSIGNMENT, [], feedback=feedback, previous=previous) == recovered


async def test_new_question_without_known_year_template_is_saved_without_guessing_numeric_replacements(setup, monkeypatch):
    experts._write(setup[3] / "plans/A-F01.json", plan_payload())
    runner = ExpertRunner(*setup, {})

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        output = question_payload(payload["case"]["question_no"])
        if output["question"]["no"] == 1:
            output["question"]["question"] = output["question"]["question"].replace("{case_year}", "2025")
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    draft = await runner.create("A", ASSIGNMENT, [])
    saved = json.loads((setup[3] / "questions/A-F01/1.json").read_text())
    assert saved["question_template"] == draft.questions[0].question
    assert "period 2025, amount 2025 USD" in saved["question_template"]
    assert "{case_year}" not in saved["question_template"]
    assert all(row.status == "draft" for row in draft.questions)


@pytest.mark.parametrize("category_options,passes", [(["Property taxes"], True), (["Income, profits and capital gains"], False), ([], False)])
async def test_planner_category_options_prevent_transfers_that_break_temporal_balance(setup, monkeypatch, category_options, passes):
    payload = plan_payload()
    setup[0]["generation"]["categories"].append({"code": "4000", "label": "Property taxes", "enabled": True})
    for case in payload["cases"]:
        case["tax_category"] = "Property taxes"
    payload["category_reallocation_reason"] = "SYNTHETIC official evidence supports the alternate category."
    experts._write(setup[3] / "plans/A-F01.json", payload)
    assignment = ASSIGNMENT | {"category": "Income, profits and capital gains", "category_options": category_options}
    runner = ExpertRunner(*setup, {})
    repeat_invalid_plan(monkeypatch, runner, payload)
    if passes:
        assert (await runner.plan("A", assignment, [])).cases[0].tax_category == "Property taxes"
    else:
        with pytest.raises(ExpertRunError, match="category_options.*temporal balance"):
            await runner.plan("A", assignment, [])
        assert not (setup[3] / "category_reallocations.json").exists()


def correction_fixture():
    from copy import deepcopy

    assignment = deepcopy(ASSIGNMENT)
    assignment["question_numbers"].append(6)
    assignment["missing_question_numbers"] = [6]
    plan = plan_payload()
    plan["cases"].append(deepcopy(plan["cases"][0]) | {
        "question_no": 6, "variant": "missing_information", "derived_from": 1,
        "missing_fact": "The second synthetic input is absent.",
    })
    output = question_payload(1)
    output["question"].update(
        no=6, variant="missing_information", question="SYNTHETIC unknown input in {case_year}.",
        answer_value="insufficient information", unit=None,
        final_answer_text="The absent second input determines the sum, so its value is needed.",
        gold_steps=[{"step": "The missing input prevents a unique sum.", "result": "insufficient information",
                     "unit": None, "citations": ["C1"]}], note="Omit the second synthetic input.",
    )
    output["citations"][0]["question_no"] = 6
    return assignment, plan, output


async def test_missing_feedback_does_not_restart_regular_writers(setup, monkeypatch):
    from copy import deepcopy

    assignment, plan, _ = correction_fixture()
    experts._write(setup[3] / "plans/A-F01.json", plan)
    runner = ExpertRunner(*setup, {})
    calls, initial_payloads = [], {}

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        number = payload["case"]["question_no"]
        calls.append((task, number, payload["feedback"]))
        initial_payloads.setdefault(number, deepcopy(payload))
        assert task == "create" and number < 6
        assert all(case["variant"] != "missing_information" for case in payload["family_plan"]["cases"])
        output = question_payload(number)
        tools.calculate("2+3")
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    first = await runner.create("A", assignment, [])
    assert [(task, number) for task, number, _ in calls] == [("create", n) for n in range(1, 6)]
    directory = setup[3] / "questions/A-F01"
    # Existing live checkpoints had the complete plan in each regular writer's
    # input and no basis_hash/task metadata. Their matching input hashes still
    # prove source and assignment identity, so a Q6 repair must preserve them.
    saved_plan = json.loads((setup[3] / "plans/A-F01.json").read_text())
    for number in range(1, 6):
        path = directory / f"{number}.json"
        saved = json.loads(path.read_text())
        saved.pop("basis_hash")
        saved.pop("task")
        prior_input = initial_payloads[number] | {"family_plan": saved_plan}
        saved["input_hash"] = hashlib.sha256(experts._json(prior_input).encode()).hexdigest()
        experts._write(path, saved)
    untouched = {n: (directory / f"{n}.json").read_bytes() for n in range(1, 6)}
    calls.clear()
    issue = "Question 6/en/C1: Supporting passage mismatch"
    corrected = await runner.create("A", assignment, [], feedback=[issue], previous=first)
    assert calls == [] and len(corrected.questions) == 5
    assert all((directory / f"{n}.json").read_bytes() == value for n, value in untouched.items())
    calls.clear()
    assert await runner.create("A", assignment, [], previous=corrected) == corrected
    assert not calls


async def test_wrong_missing_row_for_numeric_assignment_has_controlled_identity_error_before_decimal(setup, monkeypatch):
    _, plan, missing = correction_fixture()
    plan["cases"].pop()
    experts._write(setup[3] / "plans/A-F01.json", plan)
    runner = ExpertRunner(*setup, {})

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        assert payload["case"]["question_no"] == 1
        tools.calculate("2+3")
        return output_type.model_validate(missing)

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises(ExpertRunError, match="Question 1: writer changed the assigned plan identity"):
        await runner.create("A", ASSIGNMENT, [])
    assert not (setup[3] / "questions/A-F01/1.json").exists()


async def test_base_template_change_regenerates_temporal_dependencies_only(setup, monkeypatch):
    assignment, plan, _ = correction_fixture()
    experts._write(setup[3] / "plans/A-F01.json", plan)
    runner = ExpertRunner(*setup, {})
    calls = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        number = payload["case"]["question_no"]
        calls.append(number)
        output = question_payload(number)
        if number == 1 and payload["feedback"]:
            output["question"]["question"] += " Updated synthetic facts."
        elif number == 2:
            output["question"]["question"] = payload["required_question_template"]
        if task == "create":
            tools.calculate("2+3")
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    first = await runner.create("A", assignment, [])
    calls.clear()
    corrected = await runner.create("A", assignment, [], feedback=["Question 1: correct the synthetic facts"], previous=first)
    assert calls == [1, 2]
    assert corrected.questions[0].question.endswith("Updated synthetic facts.")
    assert corrected.questions[1].question.endswith("Updated synthetic facts.")
    calls.clear()
    await runner.create("A", assignment, [], feedback=["Family endpoint is ambiguous"], previous=corrected)
    assert calls == list(range(1, 6))


async def test_cached_plan_reopens_only_for_explicit_revision_feedback_and_persists_new_year(setup, monkeypatch):
    from copy import deepcopy

    path = setup[3] / "plans/A-F01.json"
    experts._write(path, plan_payload())
    runner = ExpertRunner(*setup, {})
    calls = []
    feedback = ["Question 1: the supplied synthetic edition supports 2024 instead of 2025"]

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert task == "plan" and payload["feedback"] == feedback
        assert payload["previous_plan"]["cases"][0]["case_year"] == 2025
        assert payload["source_excerpts"][0]["text"] == (setup[2] / "text.txt").read_text()
        assert payload["source_excerpts"][0]["source_id"] == "s_toy"
        revised = deepcopy(payload["previous_plan"])
        revised["cases"][0]["case_year"] = 2024
        return output_type.model_validate(revised)

    monkeypatch.setattr(runner, "_invoke", invoke)
    original = await runner.plan("A", ASSIGNMENT, [])
    assert original.cases[0].case_year == 2025 and not calls
    revised = await runner.plan("A", ASSIGNMENT, [], feedback=feedback)
    assert revised.cases[0].case_year == 2024 and calls == ["plan"]
    assert json.loads(path.read_text())["cases"][0]["case_year"] == 2024
    assert await runner.plan("A", ASSIGNMENT, []) == revised
    assert calls == ["plan"]


async def test_contradictory_positive_review_preserves_findings_and_cannot_pass_acceptance(setup, monkeypatch):
    from taxcalcbench.pipeline import review_failures

    runner = ExpertRunner(*setup, {})
    output = review_payload()
    output["questions"][0]["issues"] = ["SYNTHETIC cited rule is inapplicable to this year"]

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        tools.calculate("1+4")
        return output_type.model_validate(output)

    monkeypatch.setattr(runner, "_invoke", invoke)
    family = FamilyDraft.model_validate(family_payload())
    report = await runner.review("B", family, [])
    assert report.questions[0].passed is False
    assert report.questions[0].issues == output["questions"][0]["issues"]
    assert report.questions[0].recomputed_answer == "5.00"
    assert report.questions[0].calculation == "1+4"
    assert review_failures(family, report, "A") == ["Question 1: SYNTHETIC cited rule is inapplicable to this year"]


@pytest.mark.parametrize("long_article", [False, True])
async def test_reviewer_receives_distant_amendment_notes_and_exact_section_metadata(setup, monkeypatch, long_article):
    quote = "SYNTHETIC fixture text."
    note = "SYNTHETIC amendment note: the illustrative provision changes in a later year."
    filler = ("Unrelated synthetic article detail. " * 3 + "\n") * (450 if long_article else 12)
    article = "Article SYNTHETIC\n" + quote + "\n" + filler + note + "\n"
    following = "Next article must not replace the preceding amendment note.\n"
    text = article + following
    for filename in ("raw.txt", "text.txt"):
        (setup[2] / filename).write_text(text)
    document = setup[1]["documents"][0]
    digest = hashlib.sha256(text.encode()).hexdigest()
    gap_start = text.index(note) - 30
    document.update(sha256=digest, text_sha256=digest, partial=True,
        boundaries=[{"kind": "section", "label": "Article SYNTHETIC", "start": 0, "end": len(article)},
                    {"kind": "section", "label": "Next", "start": len(article), "end": len(text)}],
        extraction_gaps=[{"start": gap_start, "end": gap_start + 10, "label": "SYNTHETIC omitted object"}])
    runner = ExpertRunner(*setup, {})
    seen = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        seen.extend(payload["source_excerpts"])
        tools.calculate("1+4")
        return output_type.model_validate(review_payload())

    monkeypatch.setattr(runner, "_invoke", invoke)
    await runner.review("B", FamilyDraft.model_validate(family_payload()), [])
    assert len(seen) == (2 if long_article else 1)  # Five identical citations deduplicate.
    assert all(len(item["text"]) <= 40000 and following not in item["text"] for item in seen)
    assert any(quote in item["text"] for item in seen)
    amendment = next(item for item in seen if note in item["text"])
    assert amendment["end_line"] == len(article.splitlines())
    assert amendment["extraction_gaps"][0]["label"] == "SYNTHETIC omitted object"
    assert amendment["partial"] is True
    if long_article:
        assert len(article) > 40000 and len(amendment["text"]) == 4000
        start = len(article) - 4000
        assert amendment["start_line"] == text.count("\n", 0, start) + 1
    else:
        assert amendment["text"] == article and amendment["start_line"] == 1


async def test_one_family_review_receives_shared_uncited_lifecycle_context(setup, monkeypatch):
    source_dir = setup[2]
    text = "SYNTHETIC later law opening.\n" + "Body.\n" * 700 + "Repeal ends in 2026.\n"
    for name in ("later.raw.txt", "later.txt"):
        (source_dir / name).write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    later = {"id": "s_later", "title": "Synthetic later law", "kind": "law", "country": setup[0]["country"], "language": "en",
        "url": "https://official.example/later", "raw_path": "later.raw.txt", "text_path": "later.txt",
        "sha256": digest, "text_sha256": digest,
        "boundaries": [{"kind": "section", "label": "Final", "start": len(text) - 20, "end": len(text)}]}
    setup[1]["documents"].extend([later, later | {"id": "s_navigation", "kind": "navigation"},
                                  later | {"id": "s_foreign", "country": "YY"}])
    calls = []
    runner = ExpertRunner(*setup, {})

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        assert task == "review"
        assert len(payload["family"]["questions"]) == 5
        assert len(payload["source_excerpts"]) == 1  # Repeated citations are supplied once.
        assert {item["source_id"] for item in payload["lifecycle_excerpts"]} == {"s_toy", "s_later"}
        assert sum(item["source_id"] == "s_toy" for item in payload["lifecycle_excerpts"]) == 1
        assert any("ends in 2026" in item["text"] for item in payload["lifecycle_excerpts"])
        assert kwargs["supplied_source_ids"] == {"s_toy", "s_later"}
        return output_type.model_validate(review_payload())

    monkeypatch.setattr(runner, "_invoke", invoke)
    await runner.review("B", FamilyDraft.model_validate(family_payload()), [])
    assert calls == ["review"]
    assert not list(setup[3].rglob("*.source-review.json"))


def prepared_regular_for_derivation(setup, runner):
    assignment, plan, _ = correction_fixture()
    experts._write(setup[3] / "plans/A-F01.json", plan)
    regular = FamilyDraft.model_validate(family_payload())
    for row in regular.questions:
        template = row.question
        row.question = template.replace("{case_year}", str(row.case_year))
        row.status = "validated"
        experts._write(setup[3] / "questions/A-F01" / f"{row.no}.json", {
            "task": "create", "question_template": template,
            "output": {"question": row.model_dump(mode="json"),
                       "citations": [c.model_dump(mode="json") for c in regular.citations if c.question_no == row.no]}})
    runner.prompt_dir = setup[3] / "synthetic_prompts"
    runner.prompt_dir.mkdir()
    (runner.prompt_dir / "derive_missing.md").write_text("SYNTHETIC infrastructure test; return the prescribed compact edit only.")
    return assignment, regular


def compact_omission(omitted_text=", amount 2025 USD"):
    return {"omitted_text": omitted_text, "missing_fact": "The synthetic input amount.",
            "final_answer_text": "The input amount is absent, so the synthetic result cannot be computed uniquely.",
            "gold_steps": [{"step": "The omitted amount is required by the inherited synthetic calculation.",
                            "result": "insufficient information", "unit": None, "citations": ["C1"]}],
            "note": "The approved parent loses only its synthetic input amount."}


async def test_native_missing_derivation_is_one_compact_call_with_no_law_or_validation_tools(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    assignment, regular = prepared_regular_for_derivation(setup, runner)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert {item["function"]["name"] for item in body["tools"]} == {"calculate", "calculate_many"}
        payload = json.loads(next(item["content"] for item in body["messages"] if item["role"] == "user"))
        assert "sources" not in payload and "source_excerpts" not in payload
        assert payload["parent_question"]["no"] == 1 and payload["question_no"] == 6
        assert set(body["response_format"]["json_schema"]["schema"]["properties"]) == {
            "omitted_text", "missing_fact", "final_answer_text", "gold_steps", "note"}
        return completion({"role": "assistant", "content": json.dumps(compact_omission())})

    install_transport(monkeypatch, handler)
    missing = await runner.derive_missing("A", assignment, regular)
    assert len(requests) == 1 and runner.usage()["model_calls"] == 1
    assert missing is not None and len(missing.questions) == 1
    parent, derived = regular.questions[0], missing.questions[0]
    assert derived.question == parent.question.replace(", amount 2025 USD", "", 1)
    assert "period 2025" in derived.question and derived.case_year == parent.case_year
    assert derived.answer_value == "insufficient information" and derived.unit is None and derived.status == "ready"
    changes = {"no", "variant", "question", "answer_value", "unit", "status", "final_answer_text", "gold_steps", "note"}
    assert {key: value for key, value in derived.model_dump().items() if key not in changes} == {
        key: value for key, value in parent.model_dump().items() if key not in changes}
    assert [c.model_dump() for c in missing.citations] == [c.model_dump() | {"question_no": 6}
                                                         for c in regular.citations if c.question_no == parent.no]
    checkpoint = json.loads((setup[3] / "questions/A-F01/6.json").read_text())
    assert checkpoint["task"] == "derive_missing"
    assert await runner.derive_missing("A", assignment, regular) == missing
    assert len(requests) == 1
    with pytest.raises(ExpertRunError, match="regular cases only"):
        await runner.review("B", missing, [])
    assert len(requests) == 1


@pytest.mark.parametrize("problem", ["invented_span", "ambiguous_span", "year", "partial_year", "new_citation", "new_output_field"])
async def test_missing_derivation_cannot_rewrite_text_remove_year_or_invent_citations(setup, monkeypatch, problem):
    runner = ExpertRunner(*setup, {})
    assignment, regular = prepared_regular_for_derivation(setup, runner)
    result = compact_omission()
    if problem == "invented_span":
        result["omitted_text"] = "A phrase absent from the approved parent"
    elif problem == "ambiguous_span":
        path = setup[3] / "questions/A-F01/1.json"
        saved = json.loads(path.read_text())
        saved["question_template"] += " Duplicate amount 2025 USD."
        regular.questions[0].question += " Duplicate amount 2025 USD."
        experts._write(path, saved)
        result["omitted_text"] = "amount 2025 USD"
    elif problem == "year":
        result["omitted_text"] = "{case_year}"
    elif problem == "partial_year":
        result["omitted_text"] = "case_"
    elif problem == "new_citation":
        result["gold_steps"][0]["citations"] = ["C999"]
    else:
        result["citations"] = [{"citation_id": "C999"}]

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        assert task == "derive_missing"
        return output_type.model_validate(result)

    monkeypatch.setattr(runner, "_invoke", invoke)
    with pytest.raises((ExpertRunError, ValueError)):
        await runner.derive_missing("A", assignment, regular)
    assert not (setup[3] / "questions/A-F01/6.json").exists()


async def test_old_complex_missing_checkpoint_is_not_reused_and_zero_extras_needs_no_work(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    assignment, regular = prepared_regular_for_derivation(setup, runner)
    path = setup[3] / "questions/A-F01/6.json"
    experts._write(path, {"task": "missing", "input_hash": "old", "output": "not a compact derivation"})
    calls = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        return output_type.model_validate(compact_omission())

    monkeypatch.setattr(runner, "_invoke", invoke)
    await runner.derive_missing("A", assignment, regular)
    assert calls == ["derive_missing"] and json.loads(path.read_text())["task"] == "derive_missing"
    assert await runner.derive_missing("A", assignment | {"missing_question_numbers": []}, regular) is None
    assert calls == ["derive_missing"]


@pytest.mark.parametrize('repair_succeeds', [True, False])
async def test_missing_edit_repairs_changed_verb_without_rewriting_parent(setup, monkeypatch, repair_succeeds):
    runner = ExpertRunner(*setup, {})
    assignment, regular = prepared_regular_for_derivation(setup, runner)
    template = 'SYNTHETIC fixture: за {case_year} год величина составила 1 430 000 000 единиц.'
    parent = regular.questions[0]
    parent.question = template.replace('{case_year}', str(parent.case_year))
    experts._write(setup[3] / 'questions/A-F01/1.json', {'question_template': template})
    before = regular.model_dump(mode='json')
    requests = []
    incorrect = ' составляет 1 430 000 000 единиц'
    correct = ' составила 1 430 000 000 единиц'

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(next(item['content'] for item in body['messages'] if item['role'] == 'user'))
        requests.append(payload)
        assert payload['parent_question_template'] == template
        assert payload['parent_question'] == before['questions'][0]
        assert {tool['function']['name'] for tool in body['tools']} == {'calculate', 'calculate_many'}
        if len(requests) == 2:
            assert payload['previous_derivation']['omitted_text'] == incorrect
            assert 'found 0 exact matches' in payload['feedback']
            assert payload['parent_citations'] == requests[0]['parent_citations']
        assert len(requests) <= 2
        omitted = correct if len(requests) == 2 and repair_succeeds else incorrect
        return completion({'role': 'assistant', 'content': json.dumps(compact_omission(omitted))})

    install_transport(monkeypatch, handler)
    if repair_succeeds:
        result = await runner.derive_missing('A', assignment, regular)
        assert result.questions[0].question == parent.question.replace(correct, '', 1)
        assert result.questions[0].answer_value == 'insufficient information'
        assert [c.model_dump() for c in result.citations] == [c.model_dump() | {'question_no': 6}
            for c in regular.citations if c.question_no == parent.no]
        assert await runner.derive_missing('A', assignment, regular) == result
    else:
        with pytest.raises(ExpertRunError, match='Missing question 6:'):
            await runner.derive_missing('A', assignment, regular)
        assert not (setup[3] / 'questions/A-F01/6.json').exists()
    assert len(requests) == 2
    assert regular.model_dump(mode='json') == before


async def test_missing_explanations_render_year_when_reusing_a_saved_edit(setup, monkeypatch):
    runner = ExpertRunner(*setup, {})
    assignment, regular = prepared_regular_for_derivation(setup, runner)
    calls = []

    async def invoke(task, expert, payload, output_type, tools, **kwargs):
        calls.append(task)
        value = compact_omission()
        value['final_answer_text'] += ' Period {case_year}.'
        value['gold_steps'][0]['step'] += ' Period {case_year}.'
        value['note'] += ' Period {case_year}.'
        return output_type.model_validate(value)

    monkeypatch.setattr(runner, '_invoke', invoke)
    extra = await runner.derive_missing('A', assignment, regular)
    row = extra.questions[0]
    for text in (row.final_answer_text, row.gold_steps[0].step, row.note):
        assert '{case_year}' not in text and str(row.case_year) in text
    assert await runner.derive_missing('A', assignment, regular) == extra
    assert calls == ['derive_missing']


async def test_missing_variant_notes_reuse_planner_text_and_repair_old_checkpoints_offline(setup, monkeypatch):
    plan = plan_payload()
    for case in plan["cases"]:
        case["legal_difference"] = f"SYNTHETIC planner-authored difference for case {case['question_no']}."
    experts._write(setup[3] / "plans/A-F01.json", plan)
    runner = ExpertRunner(*setup, {})
    calls = []
    explicit_note = "C1: exact inception is unknown; synthetic case applicability is established."

    async def invoke(task, expert_id, payload, output_type, tools, **kwargs):
        assert task == "create"
        number = payload["case"]["question_no"]
        calls.append(number)
        result = question_payload(number)
        result["question"]["note"] = explicit_note if number == 3 else None
        return output_type.model_validate(result)

    monkeypatch.setattr(runner, "_invoke", invoke)
    family = await runner.create("A", ASSIGNMENT, [])
    assert calls == [1, 2, 3, 4, 5]
    assert family.questions[0].note is None
    assert family.questions[2].note == explicit_note
    planned_notes = {case["question_no"]: case["legal_difference"] for case in plan["cases"]}
    for number in (2, 4, 5):
        assert family.questions[number - 1].note == planned_notes[number]
        checkpoint = json.loads((setup[3] / f"questions/A-F01/{number}.json").read_text())
        assert checkpoint["output"]["question"]["note"] == planned_notes[number]

    # Previously saved writer output can omit a note too. Reuse the plan's text
    # directly, retaining explicit source uncertainty and the untouched original.
    path = setup[3] / "questions/A-F01/2.json"
    old = json.loads(path.read_text())
    old["output"]["question"]["note"] = None
    experts._write(path, old)
    unchanged = {number: (setup[3] / f"questions/A-F01/{number}.json").read_bytes() for number in (1, 3, 4, 5)}

    async def no_model(*args, **kwargs):
        pytest.fail("Filling a missing change note must not invoke a model")

    monkeypatch.setattr(runner, "_invoke", no_model)
    resumed = await runner.create("A", ASSIGNMENT, [])
    assert resumed == family
    assert json.loads(path.read_text())["output"]["question"]["note"] == planned_notes[2]
    assert all((setup[3] / f"questions/A-F01/{number}.json").read_bytes() == contents
               for number, contents in unchanged.items())

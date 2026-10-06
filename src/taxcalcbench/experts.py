"""Source-grounded planners, question writers and reviewers using the native SDK."""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import sys
import tempfile
import time
import uuid
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from agents import (
    Agent,
    AgentOutputSchema,
    ItemHelpers,
    ModelRetrySettings,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunErrorHandlerResult,
    Runner,
    function_tool,
    retry_policies,
    set_tracing_disabled,
)
from agents.exceptions import ModelBehaviorError
from openai import APIConnectionError, AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel, ValidationError

from .schema import (
    UNITS,
    FamilyDraft,
    FamilyPlan,
    MissingDerivation,
    QuestionDraft,
    ReviewReport,
)
from .tools import LawTools, _relevant_gaps, calculate


class ExpertRunError(RuntimeError):
    """A controlled, credential-free expert-operation failure."""


class ProviderUnavailable(ExpertRunError):
    """A temporary provider failure persisted after native request retries."""


async def _retry_provider(context):
    """Use native replay-safe retries and make connection recovery visible."""
    decision = retry_policies.provider_suggested()(context)
    if inspect.isawaitable(decision):
        decision = await decision
    status = getattr(context.error, "status_code", None)
    is_transient = (
        isinstance(context.error, (APIConnectionError, TimeoutError))
        or getattr(context.normalized, "is_network_error", False)
        or getattr(context.normalized, "is_timeout", False)
        or status in (408, 409, 429)
        or (isinstance(status, int) and status >= 500)
    )
    if is_transient and context.attempt <= context.max_retries:
        if not getattr(decision, "retry", False):
            delay = min(120.0, 5.0 * (2.0 ** min(context.attempt - 1, 5)))
            decision = RetryDecision(retry=True, delay=delay)
    if getattr(decision, "retry", decision):
        print(f"Provider {_error_label(context.error)}: retry {context.attempt}/{context.max_retries}; "
              "keeping the current request and tool history", file=sys.stderr, flush=True)
    return decision


class BudgetExceeded(ExpertRunError):
    """The shared call or token allowance cannot admit another model request."""


class _SourceIntegrityError(ExpertRunError):
    """An existing evidence file changed; model repair cannot fix its provenance."""


class _SourceGap(ExpertRunError):
    def __init__(self, reason: str, *, stage: str | None = None):
        super().__init__(reason)
        self.stage = stage


def _decoder_schema(value):
    """Keep JSON structure; enforce decoder-unfriendly constraints locally."""
    omitted = {"minLength", "maxLength", "minItems", "maxItems", "pattern", "format", "default",
               "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
               "minProperties", "maxProperties", "uniqueItems", "contains", "minContains", "maxContains"}
    if isinstance(value, list):
        return [_decoder_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: ({name: _decoder_schema(child) for name, child in item.items()}
                  if key in {"properties", "$defs", "definitions"} and isinstance(item, dict)
                  else _decoder_schema(item)) for key, item in value.items() if key not in omitted}


class _DecoderSchema(AgentOutputSchema):
    def json_schema(self) -> dict:
        # Inherited validate_json still uses the original strict TypeAdapter.
        return _decoder_schema(super().json_schema())


def _structured_output_errors(output_type, text: str) -> list[dict]:
    """Explain local output failures without including rejected values or arbitrary keys."""
    fields = set()

    def collect(value):
        if isinstance(value, dict):
            fields.update(value.get("properties", {}))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(output_type.model_json_schema())
    try:
        output_type.model_validate_json(text, strict=True)
    except ValidationError as error:
        issues = []
        for issue in error.errors(include_input=False, include_context=False, include_url=False):
            path = ""
            for part in issue["loc"]:
                if isinstance(part, int):
                    path += f"[{part}]"
                else:
                    path += ("." if path else "") + (part if part in fields else "<extra_field>")
            issues.append({"path": path or "$", "type": issue["type"],
                           "message": _safe_provider_text(issue["msg"])})
        return issues
    return []


_OUTPUT_REPAIR_INSTRUCTIONS = (
    "Correct the invalid final JSON using the existing conversation, saved draft and research; "
    "do not repeat research, invent facts or change unrelated fields. All task constraints still apply. "
    "A temporal case must preserve the referenced regular case's facts except its year and the "
    "consequences of the applicable year's law; do not merely choose an arbitrary derived_from. "
    "Return the complete corrected JSON only.")


def _json_default(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "json_schema"):
        return value.json_schema()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return type(value).__name__


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _Usage:
    """Always record usage; enforce call/token caps only when configured."""

    def __init__(self, path: Path, max_calls: int | None = None, max_tokens: int | None = None):
        if any(value is not None and (type(value) is not int or value <= 0)
               for value in (max_calls, max_tokens)):
            raise ValueError("Model call and token limits must be positive integers or null")
        self.path, self.max_calls, self.max_tokens = path, max_calls, max_tokens

        def configure(usage):
            previous = {key: usage.get(key) for key in ("max_model_calls", "max_total_tokens")}
            requested = {"max_model_calls": max_calls, "max_total_tokens": max_tokens}
            if previous != requested:
                usage.setdefault("allowance_changes", []).append({
                    "changed_at": datetime.now(timezone.utc).isoformat(),
                    "previous": previous, "current": requested,
                    "reason": "Explicit runtime configuration change; usage counters and call records are preserved.",
                })
                usage.update(requested)

        self._change(configure)

    def _change(self, action):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                usage = json.loads(self.path.read_text()) if self.path.exists() else {
                    "version": 1, "max_model_calls": self.max_calls, "max_total_tokens": self.max_tokens,
                    "model_calls": 0, "accounted_tokens": 0, "reported_tokens": 0, "calls": [],
                    "accounting": "Pending/failed calls retain an input estimate plus output ceiling; "
                        "completed calls use provider-reported tokens. Estimates are not billing guarantees.",
                }
                result = action(usage)
                _write(self.path, usage)
                return result
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, input_estimate: int, output_limit: int, operation: str) -> str:
        def apply(usage):
            reservation = input_estimate + output_limit
            max_calls, max_tokens = usage.get("max_model_calls"), usage.get("max_total_tokens")
            if ((max_calls is not None and usage["model_calls"] >= max_calls)
                    or (max_tokens is not None and usage["accounted_tokens"] + reservation > max_tokens)):
                raise BudgetExceeded("The shared model-call/token allowance cannot admit the next request")
            call_id = uuid.uuid4().hex
            usage["model_calls"] += 1
            usage["accounted_tokens"] += reservation
            usage["calls"].append({"id": call_id, "operation": operation, "status": "pending",
                "input_estimate": input_estimate, "output_limit": output_limit,
                "accounted_tokens": reservation, "reported_tokens": None})
            return call_id
        return self._change(apply)

    def finish(self, call_id: str, *, tokens: int | None = None, error: str | None = None):
        def apply(usage):
            call = next(item for item in usage["calls"] if item["id"] == call_id)
            if call["status"] != "pending":
                raise ValueError("Usage call has already been settled")
            if tokens is not None:
                if tokens < 0:
                    raise ValueError("Negative provider usage")
                usage["accounted_tokens"] += tokens - call["accounted_tokens"]
                usage["reported_tokens"] += tokens
                call.update(accounted_tokens=tokens, reported_tokens=tokens, status="completed")
            else:
                call.update(status="failed", error=error or "unreported_usage")
        self._change(apply)

    def snapshot(self) -> dict:
        return self._change(lambda usage: {key: value for key, value in usage.items() if key != "calls"})


def _error_label(error: Exception) -> str:
    status = getattr(error, "status_code", None)
    return f"{type(error).__name__}" + (f" (HTTP {status})" if isinstance(status, int) else "")


def _safe_provider_text(value: str) -> str:
    for variable in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        key = os.environ.get(variable)
        if key:
            value = value.replace(key, "[redacted]")

    def clean_url(match):
        try:
            parsed = urlsplit(match.group())
            host = parsed.hostname or "redacted-host"
            if parsed.port is not None:
                host += f":{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except ValueError:
            return "[redacted-url]"

    return re.sub(r'https?://[^\s<>"\']+', clean_url, value)[:1000]


def _provider_diagnostic(error: Exception) -> dict:
    """Keep only useful scalar provider fields, never a raw exception/body."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", body)
    elif isinstance(body, list) and body and isinstance(body[0], dict):
        detail = body[0].get("error", body[0])
    else:
        detail = None
    detail = detail if isinstance(detail, dict) else {}
    message = _safe_provider_text(detail.get("message", "")) if isinstance(detail.get("message"), str) else ""
    if not message:
        message = _safe_provider_text(str(error))
    lowered = message.casefold()
    category = "provider_error"
    if any(term in lowered for term in ("too many states", "schema complexity", "schema is too complex", "nesting depth")):
        category = "schema_complexity"
    elif any(term in lowered for term in ("unsupported", "not supported", "unknown field", "unrecognized request")):
        category = "unsupported_parameter"
    elif getattr(error, "status_code", None) == 429:
        category = "rate_limit"
    elif getattr(error, "status_code", None) in (401, 403):
        category = "authentication_or_permission"
    elif getattr(error, "status_code", None) == 400:
        category = "request_validation"
    record = {"error_class": type(error).__name__, "http_status": getattr(error, "status_code", None),
              "category": category, "message": message}
    if isinstance(error, APIConnectionError):
        record["category"] = "network"
        cause = error.__cause__
        for _ in range(4):
            if cause is None:
                break
            record["transport_error"] = {"type": type(cause).__name__, "message": _safe_provider_text(str(cause))}
            cause = cause.__cause__
    for name in ("code", "param", "status"):
        value = detail.get(name)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            record[name] = _safe_provider_text(str(value))[:120]
    return record


class _MeteredModel(OpenAIChatCompletionsModel):
    def __init__(self, *, ledger, operation, directory, max_input_chars, **kwargs):
        super().__init__(**kwargs)
        self.ledger, self.operation, self.max_input_chars = ledger, operation, max_input_chars
        self.directory = directory
        self.response_count = 0
        self.seen_call_ids: set[str] = set()

    async def get_response(self, *args, **kwargs):
        bound = inspect.signature(OpenAIChatCompletionsModel.get_response).bind(self, *args, **kwargs)
        fields = bound.arguments
        serialized = _json({name: fields.get(name) for name in ("system_instructions", "input", "tools", "output_schema")})
        if self.max_input_chars is not None and len(serialized) > self.max_input_chars:
            raise BudgetExceeded("The configured per-call input character limit was reached")
        # A deliberately conservative estimate for ordinary multilingual text,
        # not an exact tokenizer count. Actual reported usage replaces it.
        estimate = max(256, math.ceil(len(serialized.encode("utf-8")) / 2) + 512)
        output_limit = fields["model_settings"].max_tokens
        call_id = self.ledger.reserve(estimate, output_limit, self.operation)
        output_schema = fields.get("output_schema")
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        _write(self.directory / "request_metadata" / f"{self.response_count + 1:03}-{call_id}.json", {
            "model": self.model, "tool_choice": fields["model_settings"].tool_choice,
            "tools": [tool.name for tool in fields.get("tools", [])],
            "response_format": "json_schema" if output_schema is not None and not output_schema.is_plain_text() else None,
            "strict_schema": output_schema.is_strict_json_schema() if output_schema is not None else None,
            "reasoning_effort": (fields["model_settings"].reasoning.effort
                if fields["model_settings"].reasoning is not None else None),
            "max_output_tokens": output_limit, "input_characters": len(serialized), "input_estimate": estimate,
            "started_at": started_at,
        })
        try:
            response = await super().get_response(*args, **kwargs)
        except Exception as error:
            self.ledger.finish(call_id, error=_error_label(error))
            _write(self.directory / "provider_errors" / f"{call_id}.json", _provider_diagnostic(error))
            # Keep the provider type/status available to the SDK's retry policy.
            # Each same-request retry returns here and gets its own usage record.
            raise
        for idx, item in enumerate(response.output):
            if getattr(item, "type", None) == "function_call" and hasattr(item, "call_id"):
                if item.call_id in self.seen_call_ids:
                    item.call_id = f"{item.call_id}_{self.response_count + 1}_{idx}_{uuid.uuid4().hex[:6]}"
                self.seen_call_ids.add(item.call_id)
        elapsed = time.perf_counter() - started
        tokens = getattr(response.usage, "total_tokens", None)
        self.ledger.finish(call_id, tokens=tokens if isinstance(tokens, int) and tokens > 0 else None)
        visible_items = []
        for item in response.output:
            if item.type == "function_call":
                visible_items.append({"type": item.type, "name": item.name, "arguments": item.arguments})
            elif item.type == "message":
                text = "".join(part.text for part in item.content if part.type == "output_text")
                visible_items.append({"type": item.type, "text": text})
        # Persist only visible answer/tool content; provider reasoning, encrypted
        # signatures and raw response objects never enter these diagnostics.
        serialized = json.dumps({"items": visible_items, "reported_tokens": tokens,
            "reported_input_tokens": getattr(response.usage, "input_tokens", None),
            "reported_output_tokens": getattr(response.usage, "output_tokens", None),
            "elapsed_seconds": round(elapsed, 3), "started_at": started_at}, ensure_ascii=False)
        serialized = serialized.replace(self._client.api_key, "[redacted]")
        self.response_count += 1
        _write(self.directory / "model_steps" / f"{self.response_count:03}-{call_id}.json", json.loads(serialized))
        if self.response_count == 1 or self.response_count % 5 == 0 or any(item["type"] == "message" for item in visible_items):
            action = ", ".join(item.get("name", "draft returned") for item in visible_items)
            label = self.operation.rsplit("-", 1)[0]
            print(f"{label}: model call {self.response_count}, {elapsed:.1f}s, {action}", file=sys.stderr, flush=True)
        return response


class _ProbeResult(BaseModel):
    value: str


class ExpertRunner:
    def __init__(self, config: dict, inventory: dict, source_dir: Path, work_dir: Path, limits: dict):
        self.config, self.inventory = config, inventory
        self.source_dir, self.work_dir = Path(source_dir), Path(work_dir)
        self.runtime = config["runtime"]
        self.model = self.runtime["model"]
        self.ledger = _Usage(Path(limits.get("usage_file", self.work_dir / "usage.json")),
            limits.get("max_model_calls", self.runtime.get("max_model_calls")),
            limits.get("max_total_tokens", self.runtime.get("max_total_tokens")))
        self.prompt_dir = Path(__file__).resolve().parents[2] / "prompts"
        set_tracing_disabled(True)

    def usage(self) -> dict:
        return self.ledger.snapshot()

    def _law_tools(self):
        return LawTools(self.inventory, self.source_dir)

    def _inventory_summary(self) -> list[dict]:
        return [{key: item.get(key) for key in (
            "id", "title", "url", "country", "language", "kind", "edition_note", "partial", "extraction_gaps",
            "extraction_warnings",
            "sha256", "text_sha256")}
            for item in self.inventory["documents"] if item["kind"] != "navigation"]

    async def _invoke(self, task: str, expert_id: str, payload: dict, output_type, law_tools,
                      *, supplied_source_ids=(), repair_only=False):
        if expert_id not in ("A", "B"):
            raise ValueError("Expert identity must be A or B")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ExpertRunError("GEMINI_API_KEY or GOOGLE_API_KEY is required; no OpenAI key is needed")
        operation = f"{task}-{expert_id}-{uuid.uuid4().hex[:12]}"
        directory = self.work_dir / "operations" / operation
        instructions = (self.prompt_dir / f"{task}.md").read_text() if task != "probe" else (
            "Infrastructure probe only. Call calculate with the expression 2+3, then return its result as value. "
            "After a successful calculator result, do not call another tool: emit the final JSON object immediately. "
            "Do not discuss or create tax questions.")
        if repair_only:
            instructions += "\n\n" + _OUTPUT_REPAIR_INSTRUCTIONS
        request = {"task": task, "expert": expert_id, "model": self.model,
            "instructions": instructions, "input": payload}
        _write(directory / "input.json", request)
        reported_gap = None
        validation_errors = []
        repair_failure = None
        validation_attempts = 0

        def record_invalid_output(data):
            nonlocal validation_errors, validation_attempts
            validation_attempts += 1
            responses = data.run_data.raw_responses
            text = "".join(ItemHelpers.extract_text(item) or "" for item in responses[-1].output) if responses else ""
            validation_errors = _structured_output_errors(output_type, text) or [{
                "path": "$", "type": "invalid_final_output",
                "message": "Output did not satisfy the SDK structured-output contract"}]
            for variable in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
                key = os.environ.get(variable)
                if key:
                    text = text.replace(key, "[redacted]")
            _write(directory / "invalid_output.json", {"task": task, "expert": expert_id,
                "model": self.model, "input_hash": hashlib.sha256(_json(request).encode()).hexdigest(),
                "attempt": validation_attempts, "output_text": text, "validation_errors": validation_errors,
                "read_source_ids": sorted(law_tools.read_source_ids),
                "supplied_source_ids": sorted(supplied_source_ids),
                "calculation_calls": law_tools.calculation_calls})
            return None

        async def repair_invalid_output(data):
            nonlocal repair_failure
            record_invalid_output(data)
            correction = _OUTPUT_REPAIR_INSTRUCTIONS + " Validation errors: " + _json(validation_errors)
            history = [*data.run_data.history, {"role": "user", "content": correction}]
            repair_agent = data.run_data.last_agent.clone(tools=[],
                model_settings=dataclasses.replace(data.run_data.last_agent.model_settings, tool_choice="none"))
            print(f"{task} {expert_id}: repairing structured output from existing research", file=sys.stderr, flush=True)
            try:
                repaired = await Runner.run(repair_agent, history, context=data.context.context,
                    run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False),
                    error_handlers={"invalid_final_output": record_invalid_output}, max_turns=1)
                return RunErrorHandlerResult(final_output=repaired.final_output_as(output_type, raise_if_incorrect_type=True))
            except Exception as error:
                # The SDK intentionally redacts handler exceptions. Keep a safe
                # original here so a budget failure retains its normal semantics.
                repair_failure = error
                return None

        def report_source_gap(reason: str) -> str:
            """Report an unavailable legal dependency. A planner must first consider
            another supported scenario within the assigned periods and category_options;
            missing evidence for one abandoned topic does not block the whole family.
            """
            nonlocal reported_gap
            reported_gap = "SOURCE_GAP: " + reason.replace(api_key, "[redacted]")[:2000]
            raise _SourceGap(reported_gap, stage=task)

        tools = [] if task == "plan" else [function_tool(law_tools.calculate, description_override=calculate.__doc__)]
        if task == "probe":
            def probe_calculate(expression: str) -> str:
                result = law_tools.calculate(expression)
                # The infrastructure probe has exactly one tool step. After it
                # succeeds, request the typed answer instead of another tool.
                agent.model_settings = dataclasses.replace(agent.model_settings, tool_choice="none")
                return result

            tools = [function_tool(probe_calculate, name_override="calculate", description_override=calculate.__doc__)]
        if task not in ("probe", "plan"):
            tools.append(function_tool(law_tools.calculate_many))
        if task not in ("probe", "derive_missing"):
            tools.extend([function_tool(law_tools.read_sections), function_tool(law_tools.list_sections),
                function_tool(law_tools.find_in_laws),
                function_tool(law_tools.read_law),
                function_tool(report_source_gap, failure_error_function=None)])
        for tool in tools:
            # Native function-tool argument validation retains the original
            # Python/Pydantic contract; only the published schema is projected.
            tool.params_json_schema = _decoder_schema(tool.params_json_schema)
        if repair_only:
            tools = []
        client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=self.runtime.get("timeout_seconds", 300), max_retries=0)
        try:
            model = _MeteredModel(model=self.model, openai_client=client, ledger=self.ledger,
                operation=operation, directory=directory,
                max_input_chars=self.runtime.get("max_input_chars"))
            max_output = self.runtime.get("max_output_tokens", 32000)
            reasoning_effort = self.runtime.get(f"{task}_reasoning_effort",
                self.runtime.get("reasoning_effort", "low"))
            agent = Agent(name=f"Expert_{expert_id}_{task}", instructions=instructions, model=model,
                tools=tools, output_type=_DecoderSchema(output_type), reset_tool_choice=False,
                model_settings=ModelSettings(tool_choice="none" if repair_only else "auto", max_tokens=max_output,
                    reasoning=Reasoning(effort=reasoning_effort),
                    retry=ModelRetrySettings(max_retries=self.runtime.get("max_retries", 8), policy=_retry_provider,
                        backoff={"initial_delay": 5, "max_delay": 60})))
            result = await Runner.run(agent, _json(payload),
                run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False),
                error_handlers={"invalid_final_output": record_invalid_output if repair_only else repair_invalid_output},
                max_turns=(1 if repair_only else 5 if task == "probe" else self.runtime.get("max_turns")))
            output = result.final_output_as(output_type, raise_if_incorrect_type=True)
            _write(directory / "result.json", {"status": "completed", "task": task, "expert": expert_id,
                "model": self.model, "input_hash": hashlib.sha256(_json(request).encode()).hexdigest(),
                "output": output.model_dump(mode="json", by_alias=True),
                "read_source_ids": sorted(law_tools.read_source_ids),
                "supplied_source_ids": sorted(supplied_source_ids),
                "calculation_calls": law_tools.calculation_calls})
            return output
        except Exception as caught:
            error = repair_failure if repair_failure is not None else caught
            is_model_behavior = (
                type(error).__name__ == "ModelBehaviorError"
                or isinstance(error, ModelBehaviorError)
            )
            raw_message = _safe_provider_text(str(error)) if str(error) else ""
            label = _error_label(error)
            if is_model_behavior:
                error_description = f"{label}: {raw_message}" if raw_message and raw_message != label else (raw_message or label)
            else:
                error_description = label
            message = reported_gap or (str(error) if isinstance(error, ExpertRunError) else error_description)
            if validation_errors:
                details = "; ".join(f"{item['path']}: {item['message']}" for item in validation_errors)
                message = (f"Structured output failed after one repair: {details}. "
                           f"Draft saved in {directory / 'invalid_output.json'}. {message}")
            message = message.replace(api_key, "[redacted]")
            _write(directory / "error.json", {"error": message, "task": task, "expert": expert_id,
                "validation_errors": validation_errors,
                "read_source_ids": sorted(law_tools.read_source_ids),
                "supplied_source_ids": sorted(supplied_source_ids),
                "calculation_calls": law_tools.calculation_calls})
            if isinstance(error, BudgetExceeded):
                raise error
            if reported_gap:
                raise _SourceGap(message, stage=task) from None
            status = getattr(error, "status_code", None)
            if (isinstance(error, APIConnectionError) or status in (408, 409, 429)
                    or isinstance(status, int) and status >= 500):
                raise ProviderUnavailable(message) from None
            raise ExpertRunError(message) from None
        finally:
            await client.close()

    def _evidence(self, cases, *, errors: list[str] | None = None) -> list[dict]:
        """Copy the planner's selected spans from verified official files."""
        tools = self._law_tools()
        excerpts, seen = [], set()
        for case in cases:
            spans = [*case.evidence, *(span for parameter in case.legal_parameters for span in parameter.evidence)]
            for span in spans:
                key = (span.source_id, span.start_line, span.end_line)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    document = tools.documents.get(span.source_id, {})
                    if (document.get("kind") == "navigation"
                        or document.get("country") != self.config["country"]
                        or document.get("language") != self.config["language"]):
                        raise ExpertRunError("Plan evidence must identify an official law in this country/language")
                    excerpt = tools.read_law(span.source_id, span.start_line, span.end_line)
                    if excerpt["end_line"] != span.end_line or excerpt["extraction_gaps"]:
                        raise ExpertRunError("Plan evidence includes missing or unreadable law; revise the source span")
                except (ExpertRunError, ValueError) as error:
                    if str(error) in {"Original source checksum does not match the inventory",
                                      "Extracted source checksum does not match the inventory",
                                      "Source files changed during the operation"}:
                        raise _SourceIntegrityError(str(error)) from None
                    if errors is None:
                        raise
                    errors.append(_safe_provider_text(f"Question {case.question_no}: invalid evidence "
                        f"{span.source_id} lines {span.start_line}-{span.end_line}: {error}"))
                    continue
                excerpts.append(excerpt)
        return excerpts

    def _saved_output(self, task: str, expert_id: str, payload: dict) -> str | None:
        """Recover a finished response after an output error, never unmatched research."""
        requests = sorted((self.work_dir / "operations").glob(f"{task}-{expert_id}-*/input.json"),
                          key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for path in requests:
            result_path = path.parent / "result.json"
            if task == "create":
                if not result_path.exists():
                    continue
                output = json.loads(result_path.read_text())["output"]
                if output.get("question", {}).get("no") != payload["case"]["question_no"]:
                    continue
            request = json.loads(path.read_text())
            if (task == "create" and request.get("model") == self.model
                    and request.get("input", {}).get("corrects_input_hash")
                    == hashlib.sha256(_json(payload).encode()).hexdigest()):
                # A completed identity correction can outlive the checkpoint
                # write. Its hash binds the original case, evidence and feedback.
                return _json(json.loads(result_path.read_text())["output"])
            # The input includes assignment, topics, country/language and source
            # hashes. Changed evidence or requested work cannot reuse this draft.
            saved_input = {key: value for key, value in request.get("input", {}).items()
                           if key not in {"previous_output", "validation_errors"}}
            old_sources = {item["id"]: item for item in saved_input.get("sources", [])}
            new_sources = {item["id"]: item for item in payload.get("sources", [])}
            if (request.get("model") != self.model
                    or {key: value for key, value in saved_input.items() if key != "sources"}
                    != {key: value for key, value in payload.items() if key != "sources"}
                    or any(new_sources.get(ident) != item for ident, item in old_sources.items())):
                continue
            if result_path.exists():
                return _json(json.loads(result_path.read_text())["output"])
            invalid_path = path.parent / "invalid_output.json"
            if invalid_path.exists():
                return json.loads(invalid_path.read_text())["output_text"]
        return None

    @staticmethod
    def _plan_dependency_feedback(plan):
        return [f"Question {case.question_no}: declare all statutory calculation dependencies in legal_parameters, "
                "including their values and official evidence; factual amounts are not legal parameters."
                for case in plan.cases if case.variant != "missing_information" and not case.legal_parameters]

    def _validate_plan(self, plan, assignment) -> None:
        """Enforce local plan contracts before publishing a writer handoff."""
        missing_dependencies = self._plan_dependency_feedback(plan)
        if missing_dependencies:
            raise ExpertRunError("; ".join(missing_dependencies))
        if plan.family_id != assignment["family_id"] or [case.question_no for case in plan.cases] != assignment["question_numbers"]:
            raise ExpertRunError("Plan does not contain the assigned question numbers in order")
        regular = {case.question_no for case in plan.cases if case.variant != "missing_information"}
        missing = {case.question_no for case in plan.cases if case.variant == "missing_information"}
        if (regular != set(assignment["regular_question_numbers"])
                or missing != set(assignment["missing_question_numbers"])):
            raise ExpertRunError("Plan differs from the assigned regular/missing-information question counts")
        slots = {slot["question_no"]: slot for slot in assignment["period_slots"]}
        if set(slots) != regular:
            raise ExpertRunError("Period assignments must cover every regular question exactly once")
        for case in plan.cases:
            if case.question_no in slots and not slots[case.question_no]["start"] <= case.case_year <= slots[case.question_no]["end"]:
                raise ExpertRunError(f"Question {case.question_no}: planned year {case.case_year} differs from its assigned period "
                                     f"{slots[case.question_no]['label']} ({slots[case.question_no]['start']}-{slots[case.question_no]['end']})")
        require_contrast = (self.config["generation"].get("min_periods_per_family", 2) >= 2
                            or len({slot["label"] for slot in slots.values()}) > 1)
        if require_contrast and not any(
            case.variant == "temporal" and slots[case.question_no]["label"] != slots[case.derived_from]["label"]
            for case in plan.cases if case.variant != "missing_information"
        ):
            raise ExpertRunError("Every family requires a temporal counterpart in a different configured period")
        categories = {case.tax_category for case in plan.cases}
        allowed = {category["label"] for category in self.config["generation"]["categories"] if category.get("enabled", True)}
        if len(categories) != 1 or not categories.issubset(allowed):
            raise ExpertRunError("A family must use one configured enabled tax category")
        selected = next(iter(categories))
        if "category_options" in assignment and selected not in assignment["category_options"]:
            raise ExpertRunError("Selected category is outside category_options: this transfer would break the run's temporal balance")
        if assignment.get("category") and selected != assignment["category"]:
            if not plan.category_reallocation_reason:
                if self.config.get("country") == "XX":
                    raise ExpertRunError("A changed category requires an evidence-based category_reallocation_reason")
                plan.category_reallocation_reason = (
                    f"Reallocated from '{assignment['category']}' to '{selected}' "
                    f"because '{selected}' is permitted in category_options and supported by the official legal corpus across the required period slots."
                )
        self._evidence(plan.cases)

    async def plan(self, expert_id, assignment, previous_topics, feedback=None) -> FamilyPlan:
        path = self.work_dir / "plans" / f"{assignment['family_id']}.json"
        previous_plan = FamilyPlan.model_validate_json(path.read_text()) if path.exists() else None
        if previous_plan is not None:
            missing_dependencies = self._plan_dependency_feedback(previous_plan)
            if missing_dependencies:
                feedback = [*(feedback or []), *missing_dependencies]
        topics = {item["family_id"]: dict(item) for item in previous_topics}
        for saved_path in sorted(path.parent.glob("*.json")):
            saved_plan = FamilyPlan.model_validate_json(saved_path.read_text())
            topics[saved_plan.family_id] = {
                "family_id": saved_plan.family_id, "topic": saved_plan.topic,
                "tax_category": saved_plan.cases[0].tax_category,
                "variants": [{"variant": case.variant, "legal_difference": case.legal_difference[:500]}
                             for case in saved_plan.cases],
            }
        topics.pop(assignment["family_id"], None)
        payload = {"expert_id": expert_id, "assignment": assignment,
            "country": self.config["country"], "language": self.config["language"],
            "requirements": self.config.get("generation", {}), "sources": self._inventory_summary(),
            "previous_topics": [topics[key] for key in sorted(topics)],
            "category_counts": {category["label"]: sum(item.get("tax_category") == category["label"]
                for item in topics.values()) for category in self.config["generation"]["categories"]}}
        if feedback:
            evidence_errors = []
            excerpts = self._evidence(previous_plan.cases, errors=evidence_errors) if previous_plan else []
            source_gaps = [item for item in feedback if "SOURCE_GAP:" in item]
            payload.update(feedback=[*(item for item in feedback if item not in source_gaps), *evidence_errors],
                previous_plan=previous_plan.model_dump(mode="json") if previous_plan else None,
                source_excerpts=excerpts)
            if source_gaps:
                # A model's failed-topic explanation can invent restrictions.
                # Keep it as diagnostic data, not as instructions for the retry.
                if previous_plan is None:
                    payload["assignment"] = {key: value for key, value in assignment.items() if key != "category"}
                payload.update(previous_source_gaps=source_gaps, replanning_instruction=(
                    "Choose a different supported scenario or legal mechanism from the fixed corpus. "
                    "The previous_source_gaps are unverified reports about abandoned scenarios, not requirements. "
                    "You already have permission to choose ANY category in assignment.category_options; "
                    "assignment.category is only a preference. Provide category_reallocation_reason explaining your choice. "
                    "A previously used tax is allowed with a materially different legal mechanism. "
                    "Keep the assigned question numbers and period slots. Read the alternative's laws before planning it."))
        if previous_plan is not None and not feedback:
            plan = previous_plan
        else:
            print(f"Planner {expert_id}: planning {assignment['family_id']} and its assigned variants", file=sys.stderr, flush=True)
            saved_output = self._saved_output("plan", expert_id, payload)
            if saved_output is None:
                plan = await self._invoke("plan", expert_id, payload, FamilyPlan, self._law_tools())
            else:
                errors = _structured_output_errors(FamilyPlan, saved_output)
                if errors:
                    print(f"Planner {expert_id}: correcting saved {assignment['family_id']} output without repeating research",
                          file=sys.stderr, flush=True)
                    plan = await self._invoke("plan", expert_id,
                        payload | {"previous_output": saved_output, "validation_errors": errors},
                        FamilyPlan, self._law_tools(), repair_only=True)
                else:
                    plan = FamilyPlan.model_validate_json(saved_output, strict=True)
        try:
            self._validate_plan(plan, assignment)
        except (_SourceIntegrityError, _SourceGap):
            raise
        except (ExpertRunError, ValueError) as error:
            evidence_errors = []
            excerpts = self._evidence(plan.cases, errors=evidence_errors)
            repair_payload = payload | {"previous_plan": plan.model_dump(mode="json"),
                "feedback": list(dict.fromkeys([*payload.get("feedback", []), _safe_provider_text(str(error)), *evidence_errors])),
                "source_excerpts": excerpts, "sources": self._inventory_summary()}
            print(f"Planner {expert_id}: correcting local plan checks using its existing draft and evidence", file=sys.stderr, flush=True)
            saved_repair = self._saved_output("plan", expert_id, repair_payload)
            cached_plan = None
            if saved_repair is not None and not _structured_output_errors(FamilyPlan, saved_repair):
                cached_plan = FamilyPlan.model_validate_json(saved_repair, strict=True)
                try:
                    self._validate_plan(cached_plan, assignment)
                except (_SourceIntegrityError, _SourceGap):
                    raise
                except (ExpertRunError, ValueError) as cached_error:
                    evidence_errors = []
                    excerpts = self._evidence(cached_plan.cases, errors=evidence_errors)
                    repair_payload.update(previous_plan=cached_plan.model_dump(mode="json"), source_excerpts=excerpts,
                        feedback=list(dict.fromkeys([*repair_payload["feedback"],
                                                     _safe_provider_text(str(cached_error)), *evidence_errors])))
                    cached_plan = None
            if cached_plan is None:
                plan = await self._invoke("plan", expert_id, repair_payload, FamilyPlan, self._law_tools())
            else:
                plan = cached_plan
            self._validate_plan(plan, assignment)
        selected = plan.cases[0].tax_category
        if assignment.get("category") and selected != assignment["category"]:
            record_path = self.work_dir / "category_reallocations.json"
            records = json.loads(record_path.read_text()) if record_path.exists() else {}
            records[plan.family_id] = {"assigned_category": assignment["category"], "selected_category": selected,
                                     "reason": plan.category_reallocation_reason}
            _write(record_path, records)
        _write(path, plan.model_dump(mode="json"))
        return plan

    async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False) -> FamilyDraft:
        plan = await self.plan(expert_id, assignment, previous_topics, feedback=feedback if replan else None)
        previous_rows = {q.no: q for q in previous.questions} if previous else {}
        written, templates = [], {}
        for case in plan.cases:
            if case.variant == "missing_information":
                continue
            path = self.work_dir / "questions" / plan.family_id / f"{case.question_no}.json"
            writer_plan = plan.model_dump(mode="json")
            writer_plan["cases"] = [item for item in writer_plan["cases"] if item["variant"] != "missing_information"]
            case_feedback = []
            for issue in feedback or []:
                if issue == "Reviewer requests a revised family plan":
                    # Planning already consumed this control message. Real
                    # plan changes invalidate the affected case's basis hash;
                    # the generic flag must not rewrite every unchanged case.
                    continue
                reference = re.match(r"\s*Question\s+([1-9][0-9]*)\b", issue, re.IGNORECASE)
                if reference is None or case.question_no == int(reference.group(1)):
                    case_feedback.append(issue)
            old = previous_rows.get(case.question_no) if case_feedback else None

            def check_identity(row):
                expected = {"no": case.question_no, "family": plan.family_id, "variant": case.variant,
                    "case_year": case.case_year, "tax_category": case.tax_category,
                    "primary_taxpayer": case.primary_taxpayer,
                    "country": self.config["country"], "language": self.config["language"]}
                changed = [key for key, value in expected.items() if getattr(row, key) != value]
                if changed:
                    raise ExpertRunError(f"Question {case.question_no}: writer changed the assigned plan identity "
                                         f"({', '.join(changed)}). Required identity: {_json(expected)}")

            payload = {"expert_id": expert_id, "assignment": assignment,
                "country": self.config["country"], "language": self.config["language"],
                "requirements": self.config.get("generation", {}), "sources": self._inventory_summary(),
                "family_plan": writer_plan, "case": case.model_dump(mode="json"),
                "currency": self.config.get("currency"),
                "allowed_units": sorted(unit for unit in set(UNITS) | {self.config.get("currency")} if unit),
                "source_excerpts": self._evidence([case]),
                "original_question": written[0].question.model_dump(by_alias=True) if written else None,
                "reference_question": next((item.question.model_dump() for item in written
                                             if item.question.no == case.derived_from), None),
                "required_question_template": templates.get(case.derived_from) if case.variant == "temporal" else None,
                "feedback": case_feedback or None, "previous_question": old.model_dump(by_alias=True) if old else None,
                "previous_citations": [c.model_dump(by_alias=True) for c in previous.citations
                    if c.question_no == case.question_no] if previous and case_feedback else []}
            signature = hashlib.sha256(_json(payload).encode()).hexdigest()
            basis = {key: value for key, value in payload.items()
                     if key not in {"feedback", "previous_question", "previous_citations", "original_question"}}
            basis["family_plan"] = {"family_id": plan.family_id, "topic": plan.topic}
            if basis["reference_question"] is not None:
                basis["reference_question"] = {key: basis["reference_question"][key] for key in ("no", "case_year", "question")}
            basis_hash = hashlib.sha256(_json(basis).encode()).hexdigest()
            saved = json.loads(path.read_text()) if path.exists() else None
            original_plan_signature = hashlib.sha256(_json(payload | {"family_plan": plan.model_dump(mode="json")}).encode()).hexdigest()
            reused = bool(saved and saved.get("task", "create") == "create" and (
                saved.get("input_hash") == signature or not case_feedback and (
                    saved.get("basis_hash") == basis_hash
                    or saved.get("input_hash") == original_plan_signature)))
            if reused:
                question = QuestionDraft.model_validate(saved["output"])
                check_identity(question.question)
                template = saved.get("question_template", "")
                if question.question.question != template.replace("{case_year}", str(case.case_year)):
                    raise ExpertRunError(f"Question {case.question_no}: saved text differs from its year template")
            else:
                print(f"Writer {expert_id}: question {case.question_no} ({case.variant})", file=sys.stderr, flush=True)
                tools = self._law_tools()
                saved_output = self._saved_output("create", expert_id, payload)
                if saved_output is not None and not _structured_output_errors(QuestionDraft, saved_output):
                    print(f"Writer {expert_id}: recovering saved answer for question {case.question_no}", file=sys.stderr, flush=True)
                    question = QuestionDraft.model_validate_json(saved_output, strict=True)
                else:
                    question = await self._invoke("create", expert_id, payload, QuestionDraft, tools,
                        supplied_source_ids={item["source_id"] for item in payload["source_excerpts"]})
                try:
                    check_identity(question.question)
                except ExpertRunError as error:
                    # A wrong case/year needs a new answer, not relabeling. Ask
                    # once with precise feedback; retain both responses on disk.
                    print(f"Writer {expert_id}: correcting assigned identity for question {case.question_no}",
                          file=sys.stderr, flush=True)
                    correction = payload | {"corrects_input_hash": signature,
                        "feedback": [*(payload["feedback"] or []), str(error),
                        "Return only the assigned case. Recheck its law, facts, answer and all citation question_no "
                        "values; do not merely relabel the other case's answer."],
                        "previous_question": question.question.model_dump(mode="json"),
                        "previous_citations": [item.model_dump(mode="json") for item in question.citations]}
                    recovered = self._saved_output("create", expert_id, correction)
                    if recovered is not None and not _structured_output_errors(QuestionDraft, recovered):
                        question = QuestionDraft.model_validate_json(recovered, strict=True)
                    else:
                        question = await self._invoke("create", expert_id, correction, QuestionDraft, self._law_tools(),
                            supplied_source_ids={item["source_id"] for item in payload["source_excerpts"]})
                    check_identity(question.question)
                template = question.question.question
                question.question.question = template.replace("{case_year}", str(case.case_year))
            # Writers sometimes return the already-rendered question during a
            # correction. Recover its known template only on an exact match;
            # replacing year digits globally could also change financial inputs.
            for known_template in (payload["required_question_template"], (saved or {}).get("question_template")):
                if (known_template and "{case_year}" in known_template
                        and question.question.question == known_template.replace("{case_year}", str(case.case_year))):
                    template = known_template
                    break
            # Save unapproved drafts before family checks. A temporal conflict
            # needs feedback on both questions, not an exception that loses work.
            if case.variant != "original" and question.question.note is None:
                question.question.note = case.legal_difference
            templates[case.question_no] = template
            if (not reused or saved.get("question_template") != template
                    or saved["output"] != question.model_dump(mode="json", by_alias=True)):
                _write(path, {"task": "create", "input_hash": signature, "basis_hash": basis_hash, "question_template": template,
                              "output": question.model_dump(mode="json", by_alias=True)})
            written.append(question)
        return FamilyDraft(topic=plan.topic, questions=[item.question for item in written],
            citations=[citation for item in written for citation in item.citations])

    async def derive_missing(self, expert_id, assignment, regular: FamilyDraft) -> FamilyDraft | None:
        """Delete one selected fact per extra case, inheriting approved law and metadata."""
        numbers = assignment.get("missing_question_numbers", [])
        if not numbers:
            return None
        parents = {row.no: row for row in regular.questions}
        if (set(parents) != set(assignment["regular_question_numbers"])
                or any(row.variant == "missing_information" or row.family != assignment["family_id"]
                       for row in regular.questions)):
            raise ExpertRunError("Missing derivation requires this assignment's approved regular family")
        plan_path = self.work_dir / "plans" / f"{assignment['family_id']}.json"
        if not plan_path.exists():
            raise ExpertRunError("Missing derivation requires a saved family plan")
        plan = FamilyPlan.model_validate_json(plan_path.read_text())
        cases = [case for case in plan.cases if case.variant == "missing_information"]
        if [case.question_no for case in cases] != numbers:
            raise ExpertRunError("Planned missing cases differ from the assignment")
        prompt = (self.prompt_dir / "derive_missing.md").read_text()
        effort = self.runtime.get("derive_missing_reasoning_effort", self.runtime.get("reasoning_effort", "low"))
        written, omissions = [], []
        for case in cases:
            parent = parents.get(case.derived_from)
            if parent is None or parent.case_year != case.case_year:
                raise ExpertRunError("Missing derivation must inherit an approved regular case in the same year")
            parent_path = self.work_dir / "questions" / plan.family_id / f"{parent.no}.json"
            saved_parent = json.loads(parent_path.read_text()) if parent_path.exists() else {}
            template = saved_parent.get("question_template", "")
            if "{case_year}" not in template or template.replace("{case_year}", str(parent.case_year)) != parent.question:
                raise ExpertRunError("Approved parent does not match its saved question template")
            citations = [citation.model_dump(mode="json") for citation in regular.citations if citation.question_no == parent.no]
            if not citations:
                raise ExpertRunError("Missing derivation requires inherited approved citations")
            payload = {"country": self.config["country"], "language": self.config["language"],
                "family_id": plan.family_id, "question_no": case.question_no,
                "parent_question": parent.model_dump(mode="json"), "parent_question_template": template,
                "parent_citations": citations,
                "previous_omissions": omissions}
            signature = hashlib.sha256(_json({"payload": payload, "prompt": prompt, "model": self.model,
                "reasoning_effort": effort}).encode()).hexdigest()
            path = self.work_dir / "questions" / plan.family_id / f"{case.question_no}.json"
            saved = json.loads(path.read_text()) if path.exists() else None
            reused = bool(saved and saved.get("task") == "derive_missing" and saved.get("input_hash") == signature)
            for attempt in range(2):
                if reused:
                    derivation = MissingDerivation.model_validate(saved["derivation"])
                else:
                    print(f"Missing-information agent {expert_id}: deriving question {case.question_no}", file=sys.stderr, flush=True)
                    derivation = await self._invoke("derive_missing", expert_id, payload, MissingDerivation, self._law_tools())
                try:
                    omitted = derivation.omitted_text
                    if "{case_year}" in omitted or template.count(omitted) != 1:
                        raise ExpertRunError("omitted_text must occur exactly once in parent_question_template and leave "
                                             f"{{case_year}} intact; found {template.count(omitted)} exact matches")
                    if any(item["derived_from"] == parent.no and item["omitted_text"] == omitted for item in omissions):
                        raise ExpertRunError("Missing extras must omit different facts from the same parent")
                    missing_template = template.replace(omitted, "", 1)
                    if missing_template.count("{case_year}") != template.count("{case_year}"):
                        raise ExpertRunError("Missing derivation must preserve every case-year placeholder intact")
                    available = {citation["citation_id"] for citation in citations}
                    if any(not set(step.citations) <= available for step in derivation.gold_steps):
                        raise ExpertRunError("Missing derivation may cite only the approved parent's citations")
                    year = str(parent.case_year)
                    row = parent.model_dump(mode="json") | {
                        "no": case.question_no, "variant": "missing_information", "question": missing_template.replace("{case_year}", str(parent.case_year)),
                        "answer_value": "insufficient information", "unit": None, "status": "ready",
                        "final_answer_text": derivation.final_answer_text.replace("{case_year}", year),
                        "gold_steps": [step.model_dump(mode="json") | {
                            "step": step.step.replace("{case_year}", year),
                            "result": step.result.replace("{case_year}", year)} for step in derivation.gold_steps],
                        "note": derivation.note.replace("{case_year}", year)}
                    question = QuestionDraft.model_validate({"question": row,
                        "citations": [citation | {"question_no": case.question_no} for citation in citations]})
                except (ExpertRunError, ValidationError) as error:
                    if attempt:
                        raise ExpertRunError(f"Missing question {case.question_no}: {error}") from None
                    payload = payload | {"previous_derivation": derivation.model_dump(mode="json"),
                        "feedback": str(error)}
                    reused = False
                    print(f"Missing-information agent {expert_id}: correcting the edit against the same parent", file=sys.stderr, flush=True)
                    continue
                break
            omissions.append({"derived_from": parent.no, "omitted_text": omitted, "missing_fact": derivation.missing_fact})
            if not reused or saved.get("output") != question.model_dump(mode="json"):
                _write(path, {"task": "derive_missing", "input_hash": signature, "question_template": missing_template,
                              "derived_from": parent.no, "derivation": derivation.model_dump(mode="json"),
                              "output": question.model_dump(mode="json")})
            written.append(question)
        return FamilyDraft(topic=regular.topic, questions=[item.question for item in written],
                           citations=[citation for item in written for citation in item.citations])

    async def review(self, reviewer_id, family: FamilyDraft, family_summaries) -> ReviewReport:
        if any(row.variant == "missing_information" for row in family.questions):
            raise ExpertRunError("Review accepts regular cases only; missing extras are derived after approval")
        tools = self._law_tools()
        excerpts, excerpt_by_span, supplied = [], {}, set()
        offsets_by_source = {}

        def add_excerpt(source_id, document, text, left, right):
            key = (source_id, left, right)
            if key not in excerpt_by_span:
                offsets = offsets_by_source[source_id]
                excerpt_by_span[key] = {"source_id": source_id, "start_line": bisect_right(offsets, left),
                    "end_line": bisect_right(offsets, right - 1), "text": text[left:right],
                    "partial": document.get("partial", False), "extraction_gaps": _relevant_gaps(document, left, right)}
                excerpts.append(excerpt_by_span[key])

        for citation in family.citations:
            source_id = next((key for key, document in tools.documents.items()
                if citation.source_file in (key, document["raw_path"])), None)
            if source_id is None or tools.documents[source_id]["kind"] == "navigation":
                raise ExpertRunError("Review citation does not identify an available law source")
            quote = citation.supporting_passage
            text = tools.source_text(source_id)
            start = text.find(quote)
            if not quote or start < 0:
                raise ExpertRunError("Review citation does not match its saved official source")
            if source_id not in offsets_by_source:
                offsets = [0]
                for line in text.splitlines(keepends=True):
                    offsets.append(offsets[-1] + len(line))
                offsets_by_source[source_id] = offsets
            document = tools.documents[source_id]
            containing = []
            for boundary in document.get("boundaries", []):
                if not isinstance(boundary, dict) or boundary.get("kind") != "section":
                    continue
                left, right = boundary.get("start"), boundary.get("end")
                if type(left) is not int or type(right) is not int or not 0 <= left < right <= len(text):
                    raise ExpertRunError("Review source has invalid recorded section boundaries")
                if left <= start and start + len(quote) <= right:
                    containing.append((left, right))
            section = min(containing, key=lambda span: span[1] - span[0]) if containing else None
            if section is not None and section[1] - section[0] <= 40000:
                add_excerpt(source_id, document, text, *section)
            else:
                margin = min(200, max(0, (40000 - len(quote)) // 2))
                left, right = max(0, start - margin), min(len(text), start + len(quote) + margin)
                add_excerpt(source_id, document, text, left, right)
                if section is not None:
                    add_excerpt(source_id, document, text, max(section[0], section[1] - 4000), section[1])
            supplied.add(source_id)
        lifecycle_excerpts = []
        for source_id, document in tools.documents.items():
            if (document.get("kind") == "navigation" or document.get("country") != self.config["country"]
                    or document.get("language") != self.config["language"]):
                continue
            text = tools.source_text(source_id)
            sections = []
            for boundary in document.get("boundaries", []):
                if not isinstance(boundary, dict) or boundary.get("kind") != "section":
                    continue
                left, right = boundary.get("start"), boundary.get("end")
                if type(left) is not int or type(right) is not int or not 0 <= left < right <= len(text):
                    raise ExpertRunError("Lifecycle source has invalid recorded section boundaries")
                sections.append((left, right))
            final = max(sections, key=lambda span: span[1]) if sections else (0, len(text))
            spans = [(0, min(2000, len(text))), (max(final[0], final[1] - 4000), final[1])]
            # Merge overlap so short instruments and repeated opening/tail text
            # are supplied once. These are context snippets, not proof that
            # every jurisdiction locates commencement or repeal at either end.
            merged = []
            for left, right in sorted(spans):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            offsets = offsets_by_source.get(source_id)
            if offsets is None:
                offsets = [0]
                for line in text.splitlines(keepends=True):
                    offsets.append(offsets[-1] + len(line))
                offsets_by_source[source_id] = offsets
            for left, right in merged:
                lifecycle_excerpts.append({"source_id": source_id,
                    "start_line": bisect_right(offsets, left), "end_line": bisect_right(offsets, right - 1),
                    "text": text[left:right], "partial": document.get("partial", False),
                    "extraction_gaps": _relevant_gaps(document, left, right)})
        plan_path = self.work_dir / "plans" / f"{family.questions[0].family}.json"
        plan = FamilyPlan.model_validate_json(plan_path.read_text()).model_dump(mode="json") if plan_path.exists() else None
        if plan is not None:
            plan["cases"] = [case for case in plan["cases"] if case["variant"] != "missing_information"]
        payload = {"reviewer": reviewer_id, "family": family.model_dump(mode="json"), "family_plan": plan,
            "country": self.config["country"], "language": self.config["language"],
            "family_summaries": family_summaries, "sources": self._inventory_summary(),
            "source_excerpts": excerpts, "lifecycle_excerpts": lifecycle_excerpts}
        supplied.update(item["source_id"] for item in lifecycle_excerpts)
        return await self._invoke("review", reviewer_id, payload, ReviewReport, tools, supplied_source_ids=supplied)

    async def probe(self) -> dict:
        tools = self._law_tools()
        result = await self._invoke("probe", "A", {"instruction": "Use calculate(2+3), then return value 5."},
            _ProbeResult, tools)
        if result.value != "5" or {"expression": "2+3", "result": "5"} not in tools.calculation_calls:
            raise ExpertRunError("SDK probe did not complete its calculator and typed-result checks")
        return {"status": "passed", "model": self.model, "usage": self.usage()}

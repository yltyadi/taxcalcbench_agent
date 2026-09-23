"""Small command line interface for the local planning, writing and review workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import ROOT, read_json, run_pipeline
from .settings import build_assignments, validate_generation
from .sources import download_sources, load_inventory


def load_config(path: Path) -> dict:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Country configuration must be a JSON object")
    if not re.fullmatch(r"[A-Z]{2}", config.get("country", "")):
        raise ValueError("country must be a two-letter uppercase code")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]+)?", config.get("language", "")):
        raise ValueError("language must be a language code")
    validate_generation(config)
    for key in ("max_output_tokens", "timeout_seconds"):
        if type(config["runtime"][key]) is not int or config["runtime"][key] <= 0:
            raise ValueError(f"runtime.{key} must be a positive integer")
    if type(config["runtime"].get("max_retries", 8)) is not int or config["runtime"].get("max_retries", 8) < 0:
        raise ValueError("runtime.max_retries must be a nonnegative integer")
    for key in ("max_turns", "max_input_chars", "max_model_calls", "max_total_tokens"):
        value = config["runtime"].get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"runtime.{key} must be a positive integer or null")
    for key in ("reasoning_effort", "plan_reasoning_effort", "create_reasoning_effort", "missing_reasoning_effort", "review_reasoning_effort"):
        if config["runtime"].get(key, "low") not in {"minimal", "low", "medium", "high"}:
            raise ValueError(f"runtime.{key} must be minimal, low, medium or high")
    if not config["runtime"].get("model"):
        raise ValueError("An explicit model is required")
    if not config.get("sources"):
        raise ValueError("Official source configuration is required")
    return config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Create and independently review tax questions from official laws")
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (("download", "Download/cache official source files"),
                            ("run", "Plan families, write questions, review and export JSON"),
                            ("settings", "Show resolved counts and coverage targets without API calls"),
                            ("doctor", "Check setup; optionally test the model connection")):
        command = commands.add_parser(name, help=help_text)
        countries = command.add_mutually_exclusive_group(required=True)
        countries.add_argument("--country", type=Path, help="Country JSON configuration")
        if name == "run":
            countries.add_argument("--countries", type=Path, nargs="+",
                                   help="Run these country configurations sequentially, with separate files")
            command.add_argument("--source-root", type=Path, help="For --countries: data root (default: data)")
            command.add_argument("--output-root", type=Path, help="For --countries: output root; each country gets a subfolder")
        command.add_argument("--source-dir", type=Path, help="Default: data/<country>/sources")
        if name in ("download", "run", "settings"):
            command.add_argument("--max-documents", type=int, help="Override the configured acquisition cap; 2 is only a small download test")
        if name == "download":
            command.add_argument("--refresh", action="store_true", help="Refresh into a new source folder")
        if name == "run":
            command.add_argument("--output", type=Path, help="Required for a single country")
        if name in ("run", "settings"):
            family_count = command.add_mutually_exclusive_group()
            family_count.add_argument("--families", type=int, help="Total families for the single writer")
            family_count.add_argument("--single-family", action="store_true",
                                      help="Create and review one family using the configured regular and missing counts")
            command.add_argument("--regular-per-family", type=int, help="Regular questions in each family")
            command.add_argument("--missing-per-family", type=int, help="Additional missing-information questions per family; 0 disables")
        if name in ("run", "doctor", "settings"):
            command.add_argument("--model", help="Override the configured model without changing provider")
        if name in ("run", "doctor"):
            command.add_argument("--max-model-calls", type=int, help="Optional model-call cap")
            command.add_argument("--max-total-tokens", type=int, help="Optional accounted-token cap")
            command.add_argument("--usage-file", type=Path, help="Share usage accounting across runs")
        if name == "doctor":
            command.add_argument("--live", action="store_true", help="Spend API calls on a tool/JSON probe")
            command.add_argument("--output", type=Path, default=Path("outputs/sdk-check"))
    return result


def resolved_settings(config: dict, target: int, *, max_documents: int | None = None) -> dict:
    """A readable preview of assignments, before source applicability can change weights."""
    generation = config["generation"]
    families = build_assignments(config, target)["A"]
    category_counts = Counter(family["category"] for family in families)
    period_counts = Counter(slot["label"] for family in families for slot in family["period_slots"])
    categories = []
    for category in generation["categories"]:
        own = [family for family in families if family["category"] == category["label"]]
        categories.append({**category, "families": category_counts[category["label"]],
                           "regular_questions": sum(len(family["regular_question_numbers"]) for family in own),
                           "missing_information_questions": sum(len(family["missing_question_numbers"]) for family in own),
                           "periods": dict(Counter(slot["label"] for family in own for slot in family["period_slots"]))})
    regular = sum(len(family["regular_question_numbers"]) for family in families)
    missing = sum(len(family["missing_question_numbers"]) for family in families)
    return {"country": config["country"], "language": config["language"], "model": config["runtime"]["model"],
            "generation_blocked_reason": generation.get("blocked_reason"),
            "families": target, "regular_questions_per_family": generation.get("regular_questions_per_family", 5),
            "missing_information_per_family": generation.get("missing_information_per_family", 1),
            "min_periods_per_family": generation.get("min_periods_per_family", 2),
            "regular_questions": regular, "missing_information_questions": missing,
            "total_questions": regular + missing,
            "periods": [{**period, "regular_questions": period_counts[period["label"]]}
                        for period in generation["periods"]],
            "categories": categories,
            "source_max_documents": max_documents if max_documents is not None else config["sources"].get("max_documents"),
            "note": "Missing-information questions are additional and excluded from period ratios. "
                    "These are proposed category targets; the planner can reallocate with official-source evidence."}


async def execute(args: argparse.Namespace) -> dict:
    if getattr(args, "countries", None):
        if args.output or args.source_dir or not args.output_root:
            raise ValueError("Use --output-root and optionally --source-root with --countries, not --output/--source-dir")
        # Validate the list before any downloading or paid work; each country
        # then uses precisely the same pipeline as a single-country command.
        configs = [(path, load_config(path)) for path in args.countries]
        codes = [config["country"].lower() for _, config in configs]
        if len(codes) != len(set(codes)):
            raise ValueError("--countries must not contain duplicate country codes")
        results = []
        for path, config in configs:
            code = config["country"].lower()
            output = args.output_root / code
            country_args = argparse.Namespace(**(vars(args) | {"countries": None, "country": path,
                "source_dir": (args.source_root or ROOT / "data") / code / "sources",
                "source_root": None, "output_root": None, "output": output}))
            if config["generation"].get("blocked_reason"):
                result = {"status": "blocked", "pause_reason": config["generation"]["blocked_reason"]}
                print(f"{config['country']}: generation blocked by configuration; continuing", file=sys.stderr, flush=True)
            else:
                print(f"Starting {config['country']} ({config['language']}): {output}", file=sys.stderr, flush=True)
                try:
                    result = await execute(country_args)
                except Exception as error:
                    # Keep credentials and arbitrary provider response bodies out of logs.
                    message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
                    for variable in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
                        if os.getenv(variable):
                            message = message.replace(os.environ[variable], "[redacted]")
                    result = {"status": "error", "pause_reason": message}
                    print(f"{config['country']}: {message}; continuing to the next country",
                          file=sys.stderr, flush=True)
            results.append({key: result.get(key) for key in ("status", "pause_reason", "exported_questions",
                "requested_questions", "validated_families")} | {
                    "country": config["country"], "language": config["language"], "output": str(output)})
        return {"status": "complete" if all(row["status"] == "complete" for row in results) else "partial",
                "countries": results}
    if args.command == "run" and (not args.output or args.output_root or args.source_root):
        raise ValueError("Use --output (and optionally --source-dir) with --country")
    config = deepcopy(load_config(args.country))
    source_dir = args.source_dir or ROOT / "data" / config["country"].lower() / "sources"
    if getattr(args, "max_documents", None) is not None and args.max_documents < 1:
        raise ValueError("max-documents must be positive")
    if getattr(args, "model", None):
        config["runtime"]["model"] = args.model
    for argument, field in (("regular_per_family", "regular_questions_per_family"),
                            ("missing_per_family", "missing_information_per_family")):
        value = getattr(args, argument, None)
        if value is not None:
            config["generation"][field] = value
    validate_generation(config)
    families = getattr(args, "families", None)
    if families is not None and families < 1:
        raise ValueError("families must be positive")
    if args.command == "settings":
        target = 1 if args.single_family else (families or config["generation"]["families"])
        return resolved_settings(config, target, max_documents=args.max_documents)
    limits = {}
    for key in ("max_model_calls", "max_total_tokens"):
        value = getattr(args, key, None)
        if value is not None:
            if value < 1:
                raise ValueError(f"{key} must be positive")
            limits[key] = value
    if getattr(args, "usage_file", None):
        limits["usage_file"] = str(args.usage_file.resolve())
    if args.command == "download":
        if args.refresh and source_dir.exists() and any(source_dir.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            source_dir = source_dir.with_name(source_dir.name + "-" + stamp)
        inventory = await asyncio.to_thread(download_sources, config, source_dir,
                                            refresh=args.refresh, max_documents=args.max_documents)
        return {"status": "downloaded" if inventory["documents"] else "unavailable",
                "documents": len(inventory["documents"]), "failures": inventory.get("failures", []),
                "inventory": str(source_dir / "sources.json")}
    if args.command == "run":
        return await run_pipeline(config, source_dir, args.output,
                                  families=args.families,
                                  single_family=args.single_family,
                                  max_documents=args.max_documents, limits=limits)
    result = {"country": config["country"], "language": config["language"],
              "generation_blocked_reason": config["generation"].get("blocked_reason"),
              "model": config["runtime"]["model"], "sdk_version": version("openai-agents"),
              "gemini_key_configured": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
              "source_inventory_exists": (source_dir / "sources.json").exists(), "live_test": False}
    inventory = load_inventory(source_dir) if result["source_inventory_exists"] else {
        "country": config["country"], "language": config["language"], "documents": [], "failures": []}
    result["sources"] = {"documents": len(inventory["documents"]),
        "partial_documents": sum(bool(doc.get("partial")) for doc in inventory["documents"]),
        "failures": len(inventory["failures"]),
        "checksums_verified": result["source_inventory_exists"]}
    if args.live:
        from .experts import ExpertRunner
        runner = ExpertRunner(config, inventory, source_dir, args.output, limits)
        try:
            result["probe"] = await runner.probe()
            result["live_test"] = True
        finally:
            close = getattr(runner, "close", None)
            if close:
                await close()
    return result


def main(argv: list[str] | None = None) -> None:
    # Slurm commonly sends TERM when an allocation ends. Unwind the async run
    # just like Ctrl+C; completed checkpoints remain usable on the next node.
    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    load_dotenv(ROOT / ".env")
    args = parser().parse_args(argv)
    try:
        result = asyncio.run(execute(args))
    except KeyboardInterrupt:
        print("Interrupted; completed families remain in the output work folder", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Configuration/data error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as error:
        # Provider or HTTP response bodies may contain secrets. Leave only a controlled category on stderr.
        print(f"{type(error).__name__}: operation failed; inspect saved work records", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") in ("partial", "paused", "resource_limit", "unavailable"):
        raise SystemExit(2)

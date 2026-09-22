import json
from pathlib import Path

import pytest

from taxcalcbench import cli
from taxcalcbench.cli import load_config, parser


def test_country_defaults_and_small_command():
    config = load_config(Path("configs/kz.json"))
    assert config["language"] == "ru"
    assert config["runtime"]["model"] == "gemini-3.8-flash"
    assert config["generation"]["families"] == 50
    args = parser().parse_args(["run", "--country", "configs/kz.json", "--output", "outputs/example",
                               "--families", "1", "--max-documents", "2"])
    assert args.families == 1 and args.max_documents == 2
    assert not args.single_family
    assert config["generation"]["regular_questions_per_family"] == 5
    assert config["generation"]["missing_information_per_family"] == 1
    assert [period["weight"] for period in config["generation"]["periods"]] == [3, 1, 1]


@pytest.mark.asyncio
async def test_single_family_command_forwards_exactly_one_total_family(monkeypatch):
    recorded = {}

    async def run(*args, **kwargs):
        recorded.update(kwargs)
        return {"status": "complete", "requested_questions": 6}

    monkeypatch.setattr(cli, "run_pipeline", run)
    arguments = ["run", "--country", "configs/kz.json", "--output", "outputs/example", "--single-family"]
    args = parser().parse_args(arguments)
    result = await cli.execute(args)
    assert result["requested_questions"] == 6
    assert recorded["single_family"] is True and recorded["families"] is None
    with pytest.raises(SystemExit) as error:
        parser().parse_args(arguments + ["--families", "1"])
    assert error.value.code == 2


def test_other_country_is_configuration_only(tmp_path):
    config = load_config(Path("configs/kz.json"))
    config.update(country="XY", language="fr", currency="EUR")
    config["sources"] = {"allowed_hosts": {"official.example": ["/law"]},
                         "seeds": [{"url": "https://official.example/law", "kind": "law", "title": "Fixture"}]}
    path = tmp_path / "country.json"
    path.write_text(json.dumps(config))
    assert load_config(path)["country"] == "XY"


@pytest.mark.parametrize("change", ["count", "overlap", "duplicate"])
def test_bad_coverage_configuration_fails_before_model_use(tmp_path, change):
    config = load_config(Path("configs/kz.json"))
    generation = config["generation"]
    if change == "count":
        generation["periods"][0]["weight"] = 0
    elif change == "overlap":
        generation["periods"][1]["end"] = 2020
    else:
        generation["categories"][1]["label"] = generation["categories"][0]["label"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("field", ["max_model_calls", "max_total_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, "8"])
def test_optional_budget_limits_must_be_positive_integers(tmp_path, field, value):
    config = load_config(Path("configs/kz.json"))
    config["runtime"][field] = value
    path = tmp_path / "bad-budget.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        load_config(path)


@pytest.mark.parametrize("explicit_null", [False, True])
def test_no_default_budget_is_required(tmp_path, explicit_null):
    config = load_config(Path("configs/kz.json"))
    for key in ("max_model_calls", "max_total_tokens"):
        config["runtime"].pop(key, None)
        if explicit_null:
            config["runtime"][key] = None
    path = tmp_path / "unlimited.json"
    path.write_text(json.dumps(config))
    loaded = load_config(path)
    assert loaded["runtime"].get("max_model_calls") is None
    assert loaded["runtime"].get("max_total_tokens") is None


@pytest.mark.asyncio
async def test_explicit_cli_budget_limits_are_forwarded(monkeypatch):
    recorded = {}

    async def run(*args, **kwargs):
        recorded.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_pipeline", run)
    args = parser().parse_args(["run", "--country", "configs/kz.json", "--output", "outputs/example",
        "--single-family", "--max-model-calls", "10", "--max-total-tokens", "50000"])
    await cli.execute(args)
    assert recorded["limits"] == {"max_model_calls": 10, "max_total_tokens": 50000}


@pytest.mark.parametrize("field", ["reasoning_effort", "plan_reasoning_effort", "create_reasoning_effort", "missing_reasoning_effort", "review_reasoning_effort"])
def test_invalid_reasoning_effort_rejects_before_model_call(tmp_path, field):
    config = load_config(Path("configs/kz.json"))
    config["runtime"][field] = "unlimited"
    path = tmp_path / "bad-reasoning.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        load_config(path)


@pytest.mark.asyncio
async def test_settings_preview_has_no_source_or_model_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A settings preview must not download sources or run agents")

    monkeypatch.setattr(cli, "download_sources", forbidden)
    monkeypatch.setattr(cli, "run_pipeline", forbidden)
    result = await cli.execute(parser().parse_args(["settings", "--country", "configs/kz.json"]))
    assert result["families"] == 50
    assert result["regular_questions"] == 250
    assert result["missing_information_questions"] == 50
    assert result["total_questions"] == 300
    assert [period["regular_questions"] for period in result["periods"]] == [150, 50, 50]
    assert sorted(category["families"] for category in result["categories"]) == [8, 8, 8, 8, 9, 9]
    assert result["source_max_documents"] == 100
    for category in result["categories"]:
        assert category["periods"]["2020-2026"] == 3 * category["periods"]["2010-2019"]
        assert category["periods"]["2010-2019"] == category["periods"]["2000-2009"]


@pytest.mark.asyncio
async def test_settings_applies_count_overrides_and_excludes_missing_from_ratios():
    result = await cli.execute(parser().parse_args([
        "settings", "--country", "configs/kz.json", "--families", "25", "--regular-per-family", "6",
        "--missing-per-family", "2", "--max-documents", "8",
    ]))
    assert result["families"] == 25
    assert result["regular_questions_per_family"] == 6
    assert result["missing_information_per_family"] == 2
    assert result["regular_questions"] == 150
    assert result["missing_information_questions"] == 50
    assert [period["regular_questions"] for period in result["periods"]] == [90, 30, 30]
    assert result["source_max_documents"] == 8


@pytest.mark.asyncio
async def test_run_count_overrides_are_forwarded_without_modifying_country_file(monkeypatch):
    recorded = {}

    async def run(config, *args, **kwargs):
        recorded.update(configuration=config, **kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_pipeline", run)
    path = Path("configs/kz.json")
    before = path.read_bytes()
    await cli.execute(parser().parse_args([
        "run", "--country", str(path), "--output", "outputs/example", "--families", "75",
        "--regular-per-family", "6", "--missing-per-family", "0",
    ]))
    assert recorded["families"] == 75
    assert recorded["max_documents"] is None
    assert recorded["configuration"]["generation"]["regular_questions_per_family"] == 6
    assert recorded["configuration"]["generation"]["missing_information_per_family"] == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize(("flag", "value"), [
    ("--families", "0"), ("--regular-per-family", "1"), ("--missing-per-family", "-1"),
    ("--max-documents", "0"),
])
@pytest.mark.asyncio
async def test_invalid_cli_overrides_reject_before_pipeline(monkeypatch, flag, value):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid settings must fail before the pipeline starts")

    monkeypatch.setattr(cli, "run_pipeline", forbidden)
    args = parser().parse_args(["run", "--country", "configs/kz.json", "--output", "outputs/example", flag, value])
    with pytest.raises(ValueError):
        await cli.execute(args)


@pytest.mark.asyncio
async def test_single_family_preview_defaults_to_five_plus_one():
    result = await cli.execute(parser().parse_args(["settings", "--country", "configs/kz.json", "--single-family"]))
    assert result["total_questions"] == 6
    assert [period["regular_questions"] for period in result["periods"]] == [3, 1, 1]


@pytest.mark.asyncio
async def test_multi_country_runs_are_separate_and_partial_country_does_not_block_next(tmp_path, monkeypatch):
    paths = []
    for code, language in [("XY", "en"), ("ZZ", "pl")]:
        config = load_config(Path("configs/kz.json"))
        config.update(country=code, language=language)
        path = tmp_path / f"{code}.json"
        path.write_text(json.dumps(config))
        paths.append(str(path))
    calls = []

    async def run(config, sources, output, **kwargs):
        calls.append((config["country"], sources, output, kwargs))
        return {"status": "partial" if config["country"] == "XY" else "complete"}

    monkeypatch.setattr(cli, "run_pipeline", run)
    result = await cli.execute(parser().parse_args(["run", "--countries", *paths,
        "--source-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "results"), "--single-family"]))
    assert result["status"] == "partial" and [row["country"] for row in result["countries"]] == ["XY", "ZZ"]
    assert calls[0][1:3] == (tmp_path / "data/xy/sources", tmp_path / "results/xy")
    assert calls[1][1:3] == (tmp_path / "data/zz/sources", tmp_path / "results/zz")
    assert all(call[3]["single_family"] for call in calls)


@pytest.mark.asyncio
async def test_duplicate_country_rejects_before_pipeline(monkeypatch):
    async def forbidden(*a, **kw):
        raise AssertionError("Must reject duplicate country directories before starting")

    monkeypatch.setattr(cli, "run_pipeline", forbidden)
    with pytest.raises(ValueError, match="duplicate"):
        await cli.execute(parser().parse_args(["run", "--countries", "configs/kz.json", "configs/kz.json",
                                              "--output-root", "outputs/example"]))


@pytest.mark.parametrize("value", [-1, True, None, "8"])
def test_invalid_provider_retry_setting(tmp_path, value):
    config = load_config(Path("configs/kz.json"))
    config["runtime"]["max_retries"] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="max_retries"):
        load_config(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["eg", "in"])
async def test_batch_skips_blocked_country_without_source_or_api_work(tmp_path, monkeypatch, code):
    async def forbidden(*a, **kw):
        raise AssertionError("Blocked generation must not enter the pipeline")

    monkeypatch.setattr(cli, "run_pipeline", forbidden)
    config_path = f"configs/{code}.json"
    result = await cli.execute(parser().parse_args(["run", "--countries", config_path,
                                                    "--output-root", str(tmp_path / "out")]))
    assert result["status"] == "partial"
    assert result["countries"][0]["status"] == "blocked"
    assert result["countries"][0]["pause_reason"] == load_config(Path(config_path))["generation"]["blocked_reason"]
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_batch_continues_after_configuration_exception_and_redacts_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "SYNTHETIC_PRIVATE_KEY")
    calls = []

    async def run(config, *a, **kw):
        calls.append(config["country"])
        if config["country"] == "PK":
            raise ValueError("Synthetic source mismatch SYNTHETIC_PRIVATE_KEY")
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_pipeline", run)
    result = await cli.execute(parser().parse_args(["run", "--countries", "configs/pk.json", "configs/cn.json",
                                                    "--output-root", str(tmp_path / "out")]))
    assert calls == ["PK", "CN"] and result["countries"][1]["status"] == "complete"
    assert "source mismatch" in result["countries"][0]["pause_reason"]
    assert "SYNTHETIC_PRIVATE_KEY" not in json.dumps(result)


@pytest.mark.parametrize("code,language", [("pk", "en"), ("in", "en"), ("cn", "zh"), ("eg", "ar"), ("id", "id"), ("pl", "pl")])
def test_new_country_source_policy_and_defaults(code, language):
    from taxcalcbench.sources import _policy

    config = load_config(Path(f"configs/{code}.json"))
    policy = _policy(config)
    assert config["country"] == code.upper() and config["language"] == language
    assert config["generation"]["families"] == 50
    assert config["generation"]["regular_questions_per_family"] == 5
    assert config["generation"]["missing_information_per_family"] == 1
    assert policy["seeds"] and not policy.get("link_rules")
    assert all(seed["url"].startswith("https://") and seed["kind"] != "navigation" for seed in policy["seeds"])

"""Workflow fixtures are deliberately fictitious and never exported as benchmark data."""

import asyncio
import hashlib
import json
import shutil
from copy import deepcopy

import pytest

from taxcalcbench import pipeline
from taxcalcbench.schema import CITATION_HEADERS, QUESTION_HEADERS, FamilyDraft, ReviewReport


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    text = "NOT LAW. Fictional software fixture. Synthetic arithmetic only."
    for filename in ("fixture.html", "fixture.txt"):
        (source_dir / filename).write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    config = {"country": "XY", "language": "en", "currency": "USD", "runtime": {},
              "generation": {"families": 1,
                             "categories": [{"code": "1000", "label": "Income, profits and capital gains", "weight": 1}],
                             "regular_questions_per_family": 5, "missing_information_per_family": 0,
                             "min_periods_per_family": 2,
                             "periods": [{"label": "2020-2026", "start": 2020, "end": 2026, "weight": 3},
                                         {"label": "2010-2019", "start": 2010, "end": 2019, "weight": 1},
                                         {"label": "2000-2009", "start": 2000, "end": 2009, "weight": 1}]}}
    inventory = {"country": "XY", "language": "en", "failures": [], "documents": [
        {"id": "fixture", "title": "NOT LAW", "url": "https://official.example/fixture",
         "final_url": "https://official.example/fixture", "raw_path": "fixture.html", "text_path": "fixture.txt",
         "sha256": digest, "text_sha256": digest, "language": "en", "country": "XY", "kind": "law"}]}
    monkeypatch.setattr(pipeline, "download_sources", lambda *a, **k: deepcopy(inventory))
    return config, source_dir, tmp_path / "output", text


def family(assignment, text):
    rows, citations = [], []
    slots = {slot["question_no"]: slot for slot in assignment.get("period_slots", [])}
    for index, number in enumerate(assignment.get("regular_question_numbers", assignment["question_numbers"])):
        condition = "alpha" if index < 2 else chr(ord("a") + index) * 3
        slot = slots.get(number, next(iter(slots.values()), {"label": "2020-2026", "end": 2025}))
        variant = "original" if index == 0 else "temporal" if index == 1 else "eligibility"
        rows.append(dict(zip(QUESTION_HEADERS, [
            number, assignment["family_id"], variant, "XY", "en",
            assignment["category"] or "Income, profits and capital gains", "Individual", slot["label"], slot["end"],
            "D3", f"NOT LAW fixture {assignment['family_id']} condition {condition}, input 2 USD in {slot['end']}", "Synthetic result 5 USD",
            "5", "USD", [{"step": f"Synthetic {condition} calculation 2+3", "result": "5", "unit": "USD", "citations": ["C1"]}],
            "ready", f"Synthetic {condition} change"
        ])))
        citations.append(dict(zip(CITATION_HEADERS, [number, "en", "C1", "NOT LAW", None, "Fixture", None, None,
                         "2000-01-01", None, None, "fixture", text])))
    return FamilyDraft.model_validate({"topic": "Fixture " + assignment["family_id"], "questions": rows,
                                       "citations": citations})


def missing_extras(assignment, regular):
    rows, citations = [], []
    for index, number in enumerate(assignment["missing_question_numbers"]):
        parent = regular.questions[0 if index == 0 else index + 1]
        rows.append(parent.model_dump() | {
            "no": number, "variant": "missing_information", "status": "ready",
            "question": parent.question.replace(", input 2 USD", "", 1),
            "answer_value": "insufficient information", "unit": None,
            "final_answer_text": "The synthetic input amount is absent, so the result cannot be determined uniquely.",
            "gold_steps": [{"step": "The omitted synthetic amount is needed to determine the result.",
                            "result": "insufficient information", "unit": None, "citations": ["C1"]}],
            "note": "The approved parent loses only its synthetic input amount.",
        })
        citations.extend(citation.model_dump() | {"question_no": number}
                         for citation in regular.citations if citation.question_no == parent.no)
    return FamilyDraft.model_validate({"topic": regular.topic, "questions": rows, "citations": citations})


class FakeRunner:
    events = []
    reject = False
    always_reject = False
    text = "NOT LAW. Fictional software fixture. Synthetic arithmetic only."

    def __init__(self, *args):
        self.review_counts = {}

    async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
        self.events.append(("create", expert_id, assignment["family_id"], bool(feedback)))
        await asyncio.sleep(0)
        draft = family(assignment, self.text)
        if feedback and not self.always_reject:
            draft.questions[0].note = "Synthetic corrected clarification"
        return draft

    async def review(self, reviewer_id, draft, summaries):
        assert all(question.variant != "missing_information" for question in draft.questions)
        family_id = draft.questions[0].family
        self.events.append(("review", reviewer_id, family_id, len(summaries)))
        self.review_counts[family_id] = self.review_counts.get(family_id, 0) + 1
        fail = self.always_reject or self.reject and self.review_counts[family_id] == 1
        items = []
        for q in draft.questions:
            item = {"question_no": q.no, "passed": not fail,
                    "issues": ["Synthetic correction requested"] if fail else [],
                    "recomputed_answer": "5", "unit": "USD", "calculation": "2+3"}
            items.append(item)
        return ReviewReport.model_validate({"family_id": family_id, "reviewer": reviewer_id,
                                           "questions": items, "family_issues": []})

    async def derive_missing(self, expert_id, assignment, regular):
        assert all(question.variant != "missing_information" and question.status == "validated"
                   for question in regular.questions)
        self.events.append(("derive_missing", expert_id, assignment["family_id"], len(regular.questions)))
        return missing_extras(assignment, regular)

    def usage(self):
        return {"model_calls": 0, "reported_tokens": 0, "fixture": True}


@pytest.fixture(autouse=True)
def reset_fake():
    FakeRunner.events, FakeRunner.reject, FakeRunner.always_reject = [], False, False


@pytest.mark.asyncio
async def test_single_author_then_reviewer_exact_exports_and_resume(setup):
    config, sources, output, _ = setup
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete" and result["validated_questions"] == 5
    assert [event[0] for event in FakeRunner.events] == ["create", "review"]
    assert {(event[1], event[2]) for event in FakeRunner.events if event[0] == "review"} == {
        ("B", "A-F01")}
    rows = json.loads((output / "questions.json").read_text())
    cites = json.loads((output / "citations.json").read_text())
    assert all(list(row) == QUESTION_HEADERS and row["status"] == "validated" for row in rows)
    assert all(list(row) == CITATION_HEADERS for row in cites)
    assert len({row["no"] for row in rows}) == 5
    old_events = list(FakeRunner.events)
    repeated = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert repeated["status"] == "complete" and FakeRunner.events == old_events


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_count", [0, 1])
async def test_completed_resume_checks_saved_families_without_quadratic_export_work(setup, monkeypatch, missing_count):
    config, sources, output, _ = setup
    config["generation"].update(families=3, missing_information_per_family=missing_count)
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    before = {name: (output / name).read_bytes() for name in ("questions.json", "citations.json")}
    events = list(FakeRunner.events)
    checked = []
    original = pipeline.normalize_and_check_family

    def count_checks(*args, **kwargs):
        checked.append(kwargs["family_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "normalize_and_check_family", count_checks)
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete" and FakeRunner.events == events
    assert {name: (output / name).read_bytes() for name in before} == before
    assert set(checked) == {"A-F01", "A-F02", "A-F03"}
    assert len(checked) <= 2 * 3 * (1 + bool(missing_count))


@pytest.mark.asyncio
async def test_completed_resume_repairs_corrupt_extra_without_rewriting_regular_questions(setup):
    config, sources, output, _ = setup
    config["generation"].update(families=2, missing_information_per_family=1)
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    before = (output / "questions.json").read_bytes()
    path = output / "work/families/A-F01.json"
    state = pipeline.read_json(path)
    state["missing_draft"]["citations"][0]["supporting_passage"] = "SYNTHETIC quote absent from source"
    pipeline.save_json(path, state)
    FakeRunner.events.clear()
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete"
    assert [event[0] for event in FakeRunner.events] == ["derive_missing"]
    assert FakeRunner.events[0][2] == "A-F01"
    assert (output / "questions.json").read_bytes() == before


@pytest.mark.asyncio
async def test_normal_run_does_not_reuse_a_previous_smoke_document_cap(setup, monkeypatch):
    config, sources, output, _ = setup
    (sources / "sources.json").write_text(json.dumps({"download_limit": 2}))
    original = pipeline.download_sources
    limits = []

    def download(*args, **kwargs):
        limits.append(kwargs.get("max_documents"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "download_sources", download)
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert limits == [None]  # The downloader applies current country configuration.


@pytest.mark.asyncio
async def test_single_family_creates_five_with_a_then_b_reviews_and_resume_is_free(setup):
    config, sources, output, _ = setup
    result = await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    assert result["status"] == "complete" and result["single_family"] is True
    assert result["requested_questions"] == result["created_questions"] == result["validated_questions"] == 5
    assert result["validated_families"] == 1
    assert FakeRunner.events == [("create", "A", "A-F01", False), ("review", "B", "A-F01", 1)]
    assert all(count["coverage_required"] and not count["issues"] for count in result["coverage"].values())
    assert set(result["coverage"]) == {"A"}
    rows = json.loads((output / "questions.json").read_text())
    assert [row["no"] for row in rows] == [1, 2, 3, 4, 5]
    assert all(row["family"] == "A-F01" and row["status"] == "validated" for row in rows)
    before = list(FakeRunner.events)
    repeated = await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    assert repeated["status"] == "complete" and FakeRunner.events == before
    with pytest.raises(ValueError, match="Dataset settings changed"):
        await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)


@pytest.mark.asyncio
async def test_single_family_uses_automatic_author_correction_and_peer_review(setup):
    config, sources, output, _ = setup
    FakeRunner.reject = True
    result = await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5
    assert [event[:3] for event in FakeRunner.events] == [
        ("create", "A", "A-F01"), ("review", "B", "A-F01"),
        ("create", "A", "A-F01"), ("review", "B", "A-F01")]


@pytest.mark.asyncio
async def test_resume_after_copy_to_different_machine_path_keeps_approved_work(setup, tmp_path):
    config, sources, output, _ = setup
    await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    approved = (output / "questions.json").read_bytes()
    events = list(FakeRunner.events)
    remote = tmp_path / "hpc"
    shutil.copytree(sources, remote / "data")
    shutil.copytree(output, remote / "output")
    result = await pipeline.run_pipeline(config, remote / "data", remote / "output",
                                         single_family=True, runner_factory=FakeRunner)
    assert result["status"] == "complete" and FakeRunner.events == events
    assert (remote / "output/questions.json").read_bytes() == approved
    assert pipeline.read_json(remote / "output/work/run.json")["source_dir"] == str((remote / "data").resolve())


@pytest.mark.asyncio
async def test_relocation_does_not_bypass_changed_source_checksum(setup, tmp_path, monkeypatch):
    config, sources, output, _ = setup
    await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    inventory = pipeline.download_sources()
    inventory["documents"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(pipeline, "download_sources", lambda *a, **kw: inventory)
    with pytest.raises(ValueError, match="source documents changed"):
        await pipeline.run_pipeline(config, tmp_path / "moved-sources", output,
                                    single_family=True, runner_factory=FakeRunner)


@pytest.mark.asyncio
async def test_explicit_source_block_prevents_generation_before_download(setup, monkeypatch):
    config, sources, output, _ = setup
    config["generation"]["blocked_reason"] = "Synthetic missing historical official text"

    def forbidden(*a, **kw):
        raise AssertionError("No acquisition or model use for blocked generation")

    monkeypatch.setattr(pipeline, "download_sources", forbidden)
    with pytest.raises(ValueError, match="Synthetic missing historical"):
        await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert not output.exists()


@pytest.mark.asyncio
async def test_single_family_and_family_count_cannot_be_combined(setup):
    config, sources, output, _ = setup
    with pytest.raises(ValueError, match="either single-family or families"):
        await pipeline.run_pipeline(config, sources, output, single_family=True,
                                    families=1, runner_factory=FakeRunner)
    assert not FakeRunner.events and not output.exists()


@pytest.mark.asyncio
async def test_saved_feedback_drives_correction_then_peer_review(setup):
    config, sources, output, _ = setup
    FakeRunner.reject = True
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5
    assert sum(event[0] == "create" and event[3] for event in FakeRunner.events) == 1
    assert sum(event[0] == "review" for event in FakeRunner.events) == 2


@pytest.mark.asyncio
async def test_unresolved_families_not_exported(setup):
    config, sources, output, _ = setup
    FakeRunner.always_reject = True
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "partial" and result["validated_questions"] == 0
    assert len(result["unresolved"]) == 1
    assert sum(event[0] == "create" for event in FakeRunner.events) == 2
    assert json.loads((output / "questions.json").read_text()) == []
    assert json.loads((output / "citations.json").read_text()) == []


@pytest.mark.asyncio
async def test_resume_rejects_changed_config(setup):
    config, sources, output, _ = setup
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    config["language"] = "fr"
    with pytest.raises(ValueError, match="different country or language"):
        await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)


@pytest.mark.asyncio
async def test_bad_arithmetic_and_self_review_cannot_pass(setup):
    _, _, _, text = setup
    draft = family({"family_id": "A-F01", "question_numbers": [1, 2, 3, 4, 5], "category": None}, text)
    report = await FakeRunner().review("A", draft, [])
    report.questions[0].calculation = "100+200"
    errors = pipeline.review_failures(draft, report, "A")
    assert any("other expert" in error for error in errors)
    assert any("calculation and answer disagree" in error for error in errors)


@pytest.mark.asyncio
async def test_budget_stop_preserves_created_work(setup):
    from taxcalcbench.experts import BudgetExceeded
    config, sources, output, _ = setup

    class Limited(FakeRunner):
        async def review(self, *args):
            raise BudgetExceeded("Test allowance exhausted")

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=Limited)
    assert result["status"] == "paused"
    assert result["created_questions"] == 5 and result["validated_questions"] == 0
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5
    assert sum(event[0] == "create" for event in FakeRunner.events) == 1



@pytest.mark.asyncio
async def test_interrupted_correction_resumes_author_phase_with_original_review_feedback(setup):
    from taxcalcbench.experts import BudgetExceeded

    config, sources, output, _ = setup
    FakeRunner.reject = True

    class LimitedCorrection(FakeRunner):
        async def create(self, *args, **kwargs):
            if kwargs.get("feedback"):
                raise BudgetExceeded("Synthetic allowance exhausted during correction")
            return await super().create(*args, **kwargs)

    first = await pipeline.run_pipeline(config, sources, output, runner_factory=LimitedCorrection)
    assert first["status"] == "paused"
    states = [json.loads(path.read_text()) for path in (output / "work/families").glob("*.json")]
    assert all(state["status"] == "error" and "Synthetic correction requested" in " ".join(state["resume_issues"]) for state in states)

    class ContinueCorrection(FakeRunner):
        async def create(self, *args, **kwargs):
            assert "Synthetic correction requested" in " ".join(kwargs["feedback"])
            assert "allowance exhausted" not in " ".join(kwargs["feedback"])
            return await super().create(*args, **kwargs)

    FakeRunner.reject = False
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=ContinueCorrection)
    assert resumed["status"] == "complete" and resumed["validated_questions"] == 5
    assert sum(event[0] == "create" and event[3] for event in FakeRunner.events) == 1
    states = [json.loads(path.read_text()) for path in (output / "work/families").glob("*.json")]
    assert all(state["status"] == "validated" and not state["issues"] for state in states)


@pytest.mark.asyncio
async def test_writer_and_reviewer_share_completed_topics_across_the_batch(setup):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2

    seen = {}

    class SharedTopics(FakeRunner):
        async def create(self, expert, assignment, previous_topics, **kwargs):
            seen[assignment["family_id"]] = {item["family_id"] for item in previous_topics}
            return await super().create(expert, assignment, previous_topics, **kwargs)

        async def review(self, reviewer, draft, summaries):
            expected = {"A-F01"} if draft.questions[0].family == "A-F01" else {"A-F01", "A-F02"}
            assert {item["family_id"] for item in summaries} == expected
            return await super().review(reviewer, draft, summaries)

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=SharedTopics)
    assert result["validated_questions"] == 10
    assert "A-F01" in seen["A-F02"]
    assert seen["A-F01"] == set()


@pytest.mark.asyncio
async def test_resume_after_code_change_preserves_approved_work_without_peer_review(setup, tmp_path, monkeypatch):
    config, sources, output, _ = setup
    fake_module = tmp_path / "application/pipeline.py"
    fake_module.parent.mkdir()
    fake_module.write_text("# Synthetic implementation version one\n")
    monkeypatch.setattr(pipeline, "__file__", str(fake_module))
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    original = pipeline.read_json(output / "work/run.json")
    before = list(FakeRunner.events)
    fake_module.write_text("# Synthetic implementation version two\n")
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert resumed["validated_questions"] == 5
    assert FakeRunner.events == before
    revision = pipeline.read_json(output / "work/revisions.json")[0]
    assert revision["previous"] == original
    assert pipeline.read_json(output / "work/families/A-F01.json")["status"] == "validated"


@pytest.mark.asyncio
async def test_resume_runtime_change_keeps_original_review_provenance_without_retesting(setup):
    config, sources, output, _ = setup
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    before = list(FakeRunner.events)
    config["runtime"]["max_output_tokens"] = 32000
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5 and FakeRunner.events == before
    assert pipeline.read_json(output / "work/families/A-F01.json")["review"]


@pytest.mark.asyncio
async def test_code_change_preserves_rejection_feedback_but_requires_fresh_approval(setup, tmp_path, monkeypatch):
    config, sources, output, _ = setup
    fake_module = tmp_path / "application/pipeline.py"
    fake_module.parent.mkdir()
    fake_module.write_text("# Synthetic first version\n")
    monkeypatch.setattr(pipeline, "__file__", str(fake_module))
    FakeRunner.always_reject = True
    assert (await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner))["status"] == "partial"
    before = len(FakeRunner.events)
    fake_module.write_text("# Synthetic corrected version\n")
    FakeRunner.always_reject = False
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete"
    assert FakeRunner.events[before:] == [("create", "A", "A-F01", True), ("review", "B", "A-F01", 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["code", "additive_sources"])
@pytest.mark.parametrize("status", ["needs_correction", "correcting"])
async def test_resume_preserves_replanning_feedback_without_review_report(setup, tmp_path, monkeypatch, change, status):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2
    config["sources"] = {"allowed_hosts": {"official.example": ["/fixture"]}}
    fake_module = tmp_path / "application/pipeline.py"
    fake_module.parent.mkdir()
    fake_module.write_text("# Synthetic implementation before revision\n")
    monkeypatch.setattr(pipeline, "__file__", str(fake_module))
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    approved_path = output / "work/families/A-F01.json"
    approved = pipeline.read_json(approved_path)
    pending_path = output / "work/families/A-F02.json"
    pending = pipeline.read_json(pending_path)
    feedback = ["Synthetic eligibility facts are impossible; revise the scenario within its assigned period."]
    pending.update(status=status, review=None, issues=feedback, replan=True)
    pipeline.save_json(pending_path, pending)
    if change == "code":
        fake_module.write_text("# Synthetic implementation after revision\n")
    else:
        config["sources"]["allowed_hosts"]["official.example"].append("/additional")
    before = len(FakeRunner.events)

    class ContinueReplan(FakeRunner):
        async def create(self, expert, assignment, previous_topics, **kwargs):
            assert assignment["family_id"] == "A-F02"
            assert kwargs["feedback"] == feedback and kwargs["replan"] is True
            assert kwargs["previous"].model_dump(mode="json") == pending["draft"]
            return await super().create(expert, assignment, previous_topics, **kwargs)

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=ContinueReplan)
    assert result["status"] == "complete" and result["validated_questions"] == 10
    assert FakeRunner.events[before:] == [("create", "A", "A-F02", True), ("review", "B", "A-F02", 2)]
    assert pipeline.read_json(approved_path) == approved
    repaired = pipeline.read_json(pending_path)
    assert repaired["status"] == "validated"
    assert not repaired["issues"] and not repaired.get("replan")





@pytest.mark.asyncio
async def test_reviewer_can_request_plan_revision_before_rewriting(setup):
    config, sources, output, _ = setup
    revisions = []

    class RevisedPlan(FakeRunner):
        async def create(self, *args, replan=False, **kwargs):
            if kwargs.get("feedback"):
                revisions.append(replan)
                assert kwargs["previous"] is not None
            return await super().create(*args, **kwargs)

        async def review(self, *args):
            report = await super().review(*args)
            if self.review_counts[report.family_id] == 1:
                report.plan_revision_needed = True
                report.questions[1].passed = False
                report.questions[1].issues = ["Synthetic fixture needs another evidenced year within the same period"]
            return report

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=RevisedPlan)
    assert revisions == [True]
    assert result["status"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["generation", "sources", "currency"])
async def test_resume_still_rejects_changed_dataset_or_source_policy(setup, field):
    config, sources, output, _ = setup
    config["sources"] = {"allowed_hosts": {"official.example": ["/fixture"]}}
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    if field == "generation":
        config[field]["regular_questions_per_family"] += 1
    elif field == "sources":
        config[field]["allowed_hosts"]["official.example"] = ["/different"]
    else:
        config[field] = "EUR"
    with pytest.raises(ValueError, match="Dataset settings changed|not additive"):
        await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)


@pytest.mark.asyncio
async def test_prompt_revision_preserves_approved_exports_without_requesting_fresh_review(setup, tmp_path, monkeypatch):
    config, sources, output, _ = setup
    root = tmp_path / "application"
    (root / "prompts").mkdir(parents=True)
    (root / "references").mkdir()
    prompt = root / "prompts/review.md"
    prompt.write_text("SYNTHETIC first review prompt")
    monkeypatch.setattr(pipeline, "ROOT", root)
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert len(pipeline.read_json(output / "questions.json")) == 5
    original_rows = pipeline.read_json(output / "questions.json")
    original_citations = pipeline.read_json(output / "citations.json")
    before = list(FakeRunner.events)
    prompt.write_text("SYNTHETIC changed review prompt")
    FakeRunner.always_reject = True
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5 and FakeRunner.events == before
    assert pipeline.read_json(output / "questions.json") == original_rows
    assert pipeline.read_json(output / "citations.json") == original_citations


@pytest.mark.asyncio
async def test_modified_saved_answer_requires_correction_and_fresh_review(setup, tmp_path, monkeypatch):
    config, sources, output, _ = setup
    fake_module = tmp_path / "application/pipeline.py"
    fake_module.parent.mkdir()
    fake_module.write_text("# Synthetic implementation version one\n")
    monkeypatch.setattr(pipeline, "__file__", str(fake_module))
    await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    state_path = output / "work/families/A-F01.json"
    state = pipeline.read_json(state_path)
    state["draft"]["questions"][0]["answer_value"] = "6"
    pipeline.save_json(state_path, state)
    before = len(FakeRunner.events)
    fake_module.write_text("# Synthetic implementation version two\n")
    result = await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5 and result["status"] == "complete"
    assert FakeRunner.events[before:] == [("create", "A", "A-F01", True), ("review", "B", "A-F01", 1)]
    assert all(row["answer_value"] == "5" for row in pipeline.read_json(output / "questions.json"))


@pytest.mark.asyncio
async def test_research_only_interruption_can_resume_after_code_fix_with_revision_record(setup, tmp_path, monkeypatch):
    from taxcalcbench.experts import BudgetExceeded
    config, sources, output, _ = setup
    fake_module = tmp_path / "application/pipeline.py"
    fake_module.parent.mkdir()
    fake_module.write_text("# Synthetic implementation before fix\n")
    monkeypatch.setattr(pipeline, "__file__", str(fake_module))

    class ResearchInterrupted(FakeRunner):
        async def create(self, *args, **kwargs):
            raise BudgetExceeded("Synthetic research interruption")

    stopped = await pipeline.run_pipeline(config, sources, output, single_family=True,
                                          runner_factory=ResearchInterrupted)
    assert stopped["created_questions"] == 0
    original = pipeline.read_json(output / "work/run.json")
    fake_module.write_text("# Synthetic implementation after fix\n")
    config["runtime"]["reasoning_effort"] = "low"
    result = await pipeline.run_pipeline(config, sources, output, single_family=True, runner_factory=FakeRunner)
    assert result["validated_questions"] == 5
    revisions = pipeline.read_json(output / "work/revisions.json")
    assert len(revisions) == 1 and revisions[0]["previous"] == original
    assert pipeline.read_json(output / "work/run.json") != original


def test_coverage_counts_case_years_not_family_buckets(setup):
    config, *_ = setup
    config["generation"]["periods"] = [{"label": "recent", "start": 2020, "end": 2026, "weight": 3},
                                       {"label": "older", "start": 2010, "end": 2019, "weight": 2}]
    rows = [{"family": "A-F01", "tax_category": "Income, profits and capital gains", "case_year": year,
             "year_period_target": "recent" if year >= 2020 else "older",
             "variant": "original" if i == 0 else "temporal"} for i, year in enumerate([2026, 2025, 2020, 2018, 2017])]
    result = pipeline.coverage(rows, config, 1)
    assert result["A"]["periods"] == {"recent": 3, "older": 2} and not result["A"]["issues"]
    assert set(result) == {"A"}


@pytest.mark.asyncio
async def test_full_250_regular_plus_50_missing_target_uses_same_workflow(setup):
    from taxcalcbench.schema import CATEGORIES
    config, sources, output, _ = setup
    config["generation"].update(
        families=50,
        missing_information_per_family=1,
        categories=[{"code": code, "label": label, "weight": 1} for code, label in CATEGORIES.items()],
        periods=[{"label": "2020-2026", "start": 2020, "end": 2026, "weight": 3},
                 {"label": "2010-2019", "start": 2010, "end": 2019, "weight": 1},
                 {"label": "2000-2009", "start": 2000, "end": 2009, "weight": 1}],
    )
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete"
    assert result["validated_questions"] == 250 and result["validated_families"] == 50
    assert result["exported_questions"] == result["created_questions"] == result["requested_questions"] == 300
    assert result["generated_missing_information_questions"] == 50
    assert sum(event[0] == "create" for event in FakeRunner.events) == 50
    assert sum(event[0] == "review" for event in FakeRunner.events) == 50
    assert sum(event[0] == "derive_missing" for event in FakeRunner.events) == 50
    assert [event[0] for event in FakeRunner.events] == ["create", "review", "derive_missing"] * 50
    assert result["coverage"]["A"]["periods"] == {"2020-2026": 150, "2010-2019": 50, "2000-2009": 50}
    assert set(result["coverage"]) == {"A"}
    category_counts = result["coverage"]["A"]["categories"]
    assert set(category_counts) == set(CATEGORIES.values())
    assert max(category_counts.values()) - min(category_counts.values()) <= 1
    for category in result["coverage"]["A"]["by_category"].values():
        assert all(count > 0 for count in category["periods"].values())
        assert all(abs(count - category["period_targets"][period]) <= 1
                   for period, count in category["periods"].items())


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_count", [1, 2])
async def test_missing_information_is_derived_after_approval_and_excluded_from_period_quotas(setup, missing_count):
    config, sources, output, _ = setup
    config["generation"]["missing_information_per_family"] = missing_count
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete"
    assert result["requested_questions"] == result["exported_questions"] == result["created_questions"] == 5 + missing_count
    assert result["validated_questions"] == result["requested_regular_questions"] == 5
    assert result["generated_missing_information_questions"] == result["requested_missing_information_questions"] == missing_count
    assert FakeRunner.events == [("create", "A", "A-F01", False), ("review", "B", "A-F01", 1),
                                 ("derive_missing", "A", "A-F01", 5)]
    rows = pipeline.read_json(output / "questions.json")
    citations = pipeline.read_json(output / "citations.json")
    missing = [row for row in rows if row["variant"] == "missing_information"]
    assert len(missing) == missing_count
    assert all(row["answer_value"] == "insufficient information" and row["unit"] is None
               and row["status"] == "ready" for row in missing)
    assert all(row["status"] == "validated" for row in rows if row["variant"] != "missing_information")
    for index, row in enumerate(missing):
        parent = rows[0 if index == 0 else index + 1]
        assert [citation for citation in citations if citation["question_no"] == row["no"]] == [
            citation | {"question_no": row["no"]} for citation in citations if citation["question_no"] == parent["no"]]
        assert row["case_year"] == parent["case_year"]
    state = pipeline.read_json(output / "work/families/A-F01.json")
    assert len(state["draft"]["questions"]) == len(state["review"]["questions"]) == 5
    assert len(state["missing_draft"]["questions"]) == missing_count
    assert result["coverage"]["A"]["periods"] == {"2020-2026": 3, "2010-2019": 1, "2000-2009": 1}


@pytest.mark.asyncio
async def test_missing_content_failure_continues_and_resume_only_retries_pending_extra(setup):
    from taxcalcbench.experts import ExpertRunError

    config, sources, output, _ = setup
    config["generation"].update(families=2, missing_information_per_family=1)

    class FailedDerivation(FakeRunner):
        async def derive_missing(self, expert_id, assignment, regular):
            if assignment["family_id"] == "A-F01":
                exported = pipeline.read_json(output / "questions.json")
                assert len(exported) == 5 and all(row["status"] == "validated" for row in exported)
                self.events.append(("derive_missing_failed", expert_id, assignment["family_id"]))
                raise ExpertRunError("Synthetic omission does not match the approved parent")
            return await super().derive_missing(expert_id, assignment, regular)

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FailedDerivation)
    assert result["status"] == "partial" and result["pause_reason"] is None
    assert result["validated_questions"] == 10 and result["exported_questions"] == 11
    assert result["validated_families"] == 2 and result["generated_missing_information_questions"] == 1
    assert result["unresolved"] == [{"family_id": "A-F01", "status": "missing_information_error",
                                    "issues": ["Synthetic omission does not match the approved parent"]}]
    assert [(event[0], event[2]) for event in FakeRunner.events] == [
        ("create", "A-F01"), ("review", "A-F01"), ("derive_missing_failed", "A-F01"),
        ("create", "A-F02"), ("review", "A-F02"), ("derive_missing", "A-F02")]
    states = {path.stem: pipeline.read_json(path) for path in (output / "work/families").glob("*.json")}
    before = len(FakeRunner.events)
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert resumed["status"] == "complete" and resumed["exported_questions"] == 12
    assert FakeRunner.events[before:] == [("derive_missing", "A", "A-F01", 5)]
    for family_id, old in states.items():
        saved = pipeline.read_json(output / "work/families" / f"{family_id}.json")
        assert saved["draft"] == old["draft"] and saved["review"] == old["review"]
        assert saved["status"] == "validated" and not saved["missing_issues"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["budget", "provider", "integrity"])
async def test_missing_operational_failure_preserves_prefix_and_pauses_next_family(setup, failure):
    from taxcalcbench.experts import BudgetExceeded, ProviderUnavailable, _SourceIntegrityError

    config, sources, output, _ = setup
    config["generation"].update(families=2, missing_information_per_family=1)

    class FailedDerivation(FakeRunner):
        async def derive_missing(self, expert_id, assignment, regular):
            # The current family is approved/exported before optional derivation;
            # later families have not started yet.
            exported = pipeline.read_json(output / "questions.json")
            assert len(exported) == 5 and all(row["status"] == "validated" for row in exported)
            self.events.append(("derive_missing_failed", expert_id, assignment["family_id"]))
            if failure == "budget":
                raise BudgetExceeded("Synthetic derivation allowance exhausted")
            if failure == "provider":
                raise ProviderUnavailable("Synthetic provider outage")
            raise _SourceIntegrityError("Synthetic source checksum changed")

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FailedDerivation)
    assert result["status"] == "paused"
    assert result["validated_questions"] == result["exported_questions"] == 5
    assert result["validated_families"] == 1 and result["generated_missing_information_questions"] == 0
    assert all(item["status"] == "missing_information_error" for item in result["unresolved"])
    states = {path.stem: pipeline.read_json(path) for path in (output / "work/families").glob("*.json")}
    assert all(state["status"] == "validated" and len(state["draft"]["questions"]) == 5 for state in states.values())
    before = len(FakeRunner.events)
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert resumed["status"] == "complete" and resumed["validated_questions"] == 10
    assert resumed["exported_questions"] == 12 and resumed["generated_missing_information_questions"] == 2
    assert FakeRunner.events[before:] == [("derive_missing", "A", "A-F01", 5),
        ("create", "A", "A-F02", False), ("review", "B", "A-F02", 2), ("derive_missing", "A", "A-F02", 5)]
    for family_id, old in states.items():
        saved = pipeline.read_json(output / "work/families" / f"{family_id}.json")
        assert saved["draft"] == old["draft"] and saved["review"] == old["review"]
        assert saved["status"] == "validated" and not saved["missing_issues"]


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["no_result", "wrong_status", "wrong_number", "bad_citation"])
async def test_invalid_missing_extras_cannot_invalidate_or_replace_regular_work(setup, problem):
    config, sources, output, _ = setup
    config["generation"]["missing_information_per_family"] = 1

    class InvalidDerivation(FakeRunner):
        async def derive_missing(self, *args):
            extra = await super().derive_missing(*args)
            if problem == "no_result":
                return None
            if problem == "wrong_status":
                extra.questions[0].status = "validated"
            elif problem == "wrong_number":
                extra.questions[0].no = 99
            else:
                extra.citations[0].supporting_passage = "This invented passage is absent from the synthetic source."
            return extra

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=InvalidDerivation)
    assert result["status"] == "partial" and result["validated_questions"] == result["exported_questions"] == 5
    assert result["generated_missing_information_questions"] == 0
    assert len(result["unresolved"]) == 1 and result["unresolved"][0]["status"] == "missing_information_error"
    assert [row["no"] for row in pipeline.read_json(output / "questions.json")] == [1, 2, 3, 4, 5]
    assert sum(event[0] == "review" for event in FakeRunner.events) == 1

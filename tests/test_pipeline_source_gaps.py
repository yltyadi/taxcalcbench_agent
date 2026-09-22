"""Unsupported scenarios replan against a fixed corpus and retain resume feedback."""

import asyncio
from copy import deepcopy

import pytest
from test_pipeline import FakeRunner, family
from test_pipeline import reset_fake as reset_fake
from test_pipeline import setup as setup

from taxcalcbench import pipeline
from taxcalcbench.experts import _SourceGap

GAP = "SOURCE_GAP: SYNTHETIC historical parameter evidence unavailable"


def mutable_inventory(setup, monkeypatch):
    config, sources, _, _ = setup
    inventory = pipeline.download_sources(config, sources)
    monkeypatch.setattr(pipeline, "download_sources", lambda *args, **kwargs: deepcopy(inventory))
    return inventory


def append_source(inventory, source_dir):
    document = deepcopy(inventory["documents"][0])
    for key, filename in (("raw_path", "index.html"), ("text_path", "index.txt")):
        (source_dir / filename).write_bytes((source_dir / document[key]).read_bytes())
        document[key] = filename
    document.update(id="added-index", title="SYNTHETIC additional official index",
                    url="https://official.example/index", final_url="https://official.example/index")
    inventory["documents"].append(document)


async def test_writer_source_gap_replans_with_original_feedback_without_downloading_again(setup, monkeypatch):
    config, sources, output, _ = setup
    calls, downloads = [], []
    original_download = pipeline.download_sources

    def download(*args, **kwargs):
        downloads.append(True)
        return original_download(*args, **kwargs)

    monkeypatch.setattr(pipeline, "download_sources", download)

    class Replans(FakeRunner):
        async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
            calls.append((replan, feedback, previous))
            if len(calls) == 1:
                raise _SourceGap(GAP, stage="create")
            assert replan and feedback == [GAP] and previous is None
            return family(assignment, self.text)

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=Replans)
    assert result["status"] == "complete" and result["validated_questions"] == 5
    assert len(calls) == 2 and not calls[0][0] and len(downloads) == 1
    state = pipeline.read_json(output / "work/families/A-F01.json")
    assert state["status"] == "validated" and not state["replan"] and not state["issues"]


@pytest.mark.parametrize("stage, expected_calls", [("plan", 1), ("create", 2)])
async def test_confirmed_or_repeated_gap_remains_pending_without_spinning(setup, stage, expected_calls):
    config, sources, output, _ = setup
    calls = []

    class Missing(FakeRunner):
        async def create(self, *args, **kwargs):
            calls.append(kwargs.get("replan", False))
            assert len(calls) <= expected_calls
            raise _SourceGap(GAP, stage=stage)

    result = await asyncio.wait_for(
        pipeline.run_pipeline(config, sources, output, runner_factory=Missing), timeout=2)
    state = pipeline.read_json(output / "work/families/A-F01.json")
    assert result["status"] == "partial" and state["status"] == "needs_correction"
    assert state["replan"] and state["issues"] == [GAP]
    assert len(calls) == expected_calls and all(calls[1:])
    assert not [event for event in FakeRunner.events if event[0] == "review"]


async def test_reviewer_source_gap_replans_and_rechecks_within_the_same_run(setup):
    config, sources, output, _ = setup
    create_calls, review_calls = [], []

    class Repairs(FakeRunner):
        async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
            create_calls.append(replan)
            if replan:
                assert feedback == [GAP] and previous is not None
            return family(assignment, self.text)

        async def review(self, *args):
            review_calls.append(True)
            if len(review_calls) == 1:
                raise _SourceGap(GAP, stage="review")
            return await super().review(*args)

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=Repairs)
    assert result["status"] == "complete"
    assert create_calls == [False, True] and len(review_calls) == 2


@pytest.mark.parametrize("failure_stage", ["review", "create", "plan"])
async def test_repeated_review_gap_stops_even_when_the_rewrite_also_fails(setup, failure_stage):
    config, sources, output, _ = setup
    creates, reviews = [], []

    class NeverResolved(FakeRunner):
        async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
            creates.append(replan)
            assert len(creates) <= 2
            if replan and failure_stage in {"create", "plan"}:
                raise _SourceGap(GAP, stage=failure_stage)
            return family(assignment, self.text)

        async def review(self, *args):
            reviews.append(True)
            assert len(reviews) <= 2
            raise _SourceGap(GAP, stage="review")

    result = await asyncio.wait_for(
        pipeline.run_pipeline(config, sources, output, runner_factory=NeverResolved), timeout=2)
    state = pipeline.read_json(output / "work/families/A-F01.json")
    assert result["status"] == "partial" and state["status"] == "needs_correction"
    assert len(reviews) == (2 if failure_stage == "review" else 1)
    assert creates == [False, True]


async def test_added_sources_between_runs_resume_gap_and_preserve_validated_family(setup, monkeypatch):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2
    inventory = mutable_inventory(setup, monkeypatch)
    calls = []

    class SourceDependent(FakeRunner):
        def __init__(self, config, inventory, *args):
            super().__init__()
            self.has_index = any(row["id"] == "added-index" for row in inventory["documents"])

        async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
            calls.append((assignment["family_id"], replan))
            if assignment["family_id"] == "A-F02" and not self.has_index:
                raise _SourceGap(GAP, stage="plan")
            if self.has_index:
                assert assignment["family_id"] == "A-F02" and replan and feedback == [GAP]
            return family(assignment, self.text)

    first = await pipeline.run_pipeline(config, sources, output, runner_factory=SourceDependent)
    assert first["status"] == "partial" and first["validated_questions"] == 5
    approved_path = output / "work/families/A-F01.json"
    approved_state = approved_path.read_bytes()
    approved_rows = pipeline.read_json(output / "questions.json")
    append_source(inventory, sources)
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=SourceDependent)
    assert result["status"] == "complete" and result["validated_questions"] == 10
    assert calls == [("A-F01", False), ("A-F02", False), ("A-F02", True)]
    assert approved_path.read_bytes() == approved_state
    assert [row for row in pipeline.read_json(output / "questions.json") if row["family"] == "A-F01"] == approved_rows
    assert len([event for event in FakeRunner.events if event[0] == "review" and event[2] == "A-F01"]) == 1


async def test_additive_official_policy_and_document_resume_preserves_all_approvals(setup, monkeypatch):
    config, sources, output, _ = setup
    config["sources"] = {"allowed_hosts": {"official.example": ["/fixture"]}}
    inventory = mutable_inventory(setup, monkeypatch)
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    previous_events = list(FakeRunner.events)
    previous_state = (output / "work/families/A-F01.json").read_bytes()
    previous_exports = (output / "questions.json").read_bytes()
    config["sources"]["allowed_hosts"]["official.example"].append("/index")
    append_source(inventory, sources)
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete" and FakeRunner.events == previous_events
    assert (output / "work/families/A-F01.json").read_bytes() == previous_state
    assert (output / "questions.json").read_bytes() == previous_exports


@pytest.mark.parametrize("change", ["raw_hash", "text_hash", "remove"])
async def test_existing_source_changes_or_removal_reject_resume_before_mutating_outputs(setup, monkeypatch, change):
    config, sources, output, _ = setup
    inventory = mutable_inventory(setup, monkeypatch)
    append_source(inventory, sources)
    await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    previous_exports = (output / "questions.json").read_bytes()
    previous_signature = (output / "work/run.json").read_bytes()
    previous_events = list(FakeRunner.events)
    if change == "remove":
        inventory["documents"].pop(0)
    else:
        inventory["documents"][0]["sha256" if change == "raw_hash" else "text_sha256"] = "changed"
    with pytest.raises(ValueError, match="Existing source documents changed or were removed"):
        await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert FakeRunner.events == previous_events
    assert (output / "questions.json").read_bytes() == previous_exports
    assert (output / "work/run.json").read_bytes() == previous_signature


@pytest.mark.parametrize("during_review", [False, True])
async def test_pending_source_gap_resumes_with_original_feedback(setup, during_review):
    config, sources, output, _ = setup
    recovered, calls = False, []

    class Recoverable(FakeRunner):
        async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
            calls.append({"replan": replan, "feedback": feedback, "has_previous": previous is not None})
            if not during_review and not recovered:
                raise _SourceGap(GAP, stage="create")
            return family(assignment, self.text)

        async def review(self, *args):
            if not recovered:
                raise _SourceGap(GAP, stage="review")
            return await super().review(*args)

    first = await pipeline.run_pipeline(config, sources, output, runner_factory=Recoverable)
    state_path = output / "work/families/A-F01.json"
    failed = pipeline.read_json(state_path)
    assert first["status"] == "partial" and failed["status"] == "needs_correction"
    assert bool(failed["draft"]) is during_review and failed["issues"] == [GAP]
    recovered = True
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=Recoverable)
    assert resumed["status"] == "complete" and resumed["validated_questions"] == 5
    assert calls[-1] == {"replan": True, "feedback": failed["issues"], "has_previous": during_review}
    assert pipeline.read_json(state_path)["status"] == "validated"

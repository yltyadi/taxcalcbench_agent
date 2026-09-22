"""Synthetic integration checks for the sequential, fixed-corpus workflow."""

import json

import pytest
from test_pipeline import FakeRunner
from test_pipeline import setup as pipeline_setup

from taxcalcbench import pipeline
from taxcalcbench.experts import ProviderUnavailable

setup = pipeline_setup


@pytest.fixture(autouse=True)
def reset_events():
    FakeRunner.events, FakeRunner.reject, FakeRunner.always_reject = [], False, False


async def test_finish_each_family_before_starting_next_and_prepare_sources_once(setup, monkeypatch):
    config, sources, output, _ = setup
    config["generation"].update(families=2, missing_information_per_family=1)
    original = pipeline.download_sources
    downloads = []

    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "download_sources", download)
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "complete"
    assert result["validated_questions"] == 10
    assert result["generated_missing_information_questions"] == 2
    assert [(event[0], event[2]) for event in FakeRunner.events] == [
        (stage, family) for family in ("A-F01", "A-F02")
        for stage in ("create", "review", "derive_missing")]
    assert len(downloads) == 1


async def test_persistent_provider_failure_preserves_approval_and_stops_next_family(setup):
    config, sources, output, _ = setup
    config["generation"]["families"] = 3

    class InterruptedRunner(FakeRunner):
        async def create(self, expert_id, assignment, *args, **kwargs):
            if assignment["family_id"] == "A-F02":
                self.events.append(("provider_error", expert_id, "A-F02", False))
                raise ProviderUnavailable("InternalServerError (HTTP 500)")
            return await super().create(expert_id, assignment, *args, **kwargs)

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=InterruptedRunner)
    assert result["status"] == "paused"
    assert result["validated_questions"] == 5
    assert "HTTP 500" in result["pause_reason"]
    assert {event[2] for event in FakeRunner.events} == {"A-F01", "A-F02"}
    assert not (output / "work/families/A-F03.json").exists()
    assert {row["family"] for row in json.loads((output / "questions.json").read_text())} == {"A-F01"}
    approved = (output / "work/families/A-F01.json").read_bytes()
    FakeRunner.events.clear()
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert resumed["status"] == "complete"
    assert {event[2] for event in FakeRunner.events} == {"A-F02", "A-F03"}
    assert (output / "work/families/A-F01.json").read_bytes() == approved


async def test_changed_drafts_can_be_corrected_beyond_six_attempts_without_a_setting(setup):
    config, sources, output, _ = setup

    class ImprovingRunner(FakeRunner):
        def __init__(self, *args):
            super().__init__(*args)
            self.attempt = 0

        async def create(self, *args, **kwargs):
            self.attempt += 1
            draft = await super().create(*args, **kwargs)
            draft.questions[0].final_answer_text = f"Synthetic worked explanation revision {self.attempt}: 5 USD."
            return draft

        async def review(self, *args, **kwargs):
            report = await super().review(*args, **kwargs)
            if self.attempt < 8:
                report.questions[0].passed = False
                report.questions[0].issues = [f"Synthetic explanation defect {self.attempt}"]
            return report

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=ImprovingRunner)
    assert result["status"] == "complete"
    assert sum(event[0] == "create" for event in FakeRunner.events) == 8
    assert sum(event[0] == "review" for event in FakeRunner.events) == 8


async def test_unchanged_rejected_draft_does_not_loop_or_get_exported(setup):
    config, sources, output, _ = setup
    FakeRunner.always_reject = True
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=FakeRunner)
    assert result["status"] == "partial"
    assert result["validated_questions"] == 0
    assert [event[0] for event in FakeRunner.events] == ["create", "review", "create"]
    assert json.loads((output / "questions.json").read_text()) == []
    assert result["unresolved"][0]["issues"]


async def test_note_only_citation_uncertainty_correction_gets_reviewed(setup):
    config, sources, output, _ = setup

    class NoteCorrectionRunner(FakeRunner):
        def __init__(self, *args):
            super().__init__(*args)
            self.attempt = 0

        async def create(self, *args, **kwargs):
            self.attempt += 1
            draft = await super().create(*args, **kwargs)
            if self.attempt > 1:
                draft.questions[0].note = "C1: exact inception is unknown; this scenario's applicability is established."
            return draft

        async def review(self, *args, **kwargs):
            report = await super().review(*args, **kwargs)
            if self.attempt == 1:
                report.questions[0].passed = False
                report.questions[0].issues = ["Question 1: explain C1's unknown boundary in note"]
            return report

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=NoteCorrectionRunner)
    assert result["status"] == "complete"
    assert [event[0] for event in FakeRunner.events] == ["create", "review", "create", "review"]


async def test_missing_export_uses_normalized_official_source_bindings(setup):
    config, sources, output, _ = setup
    config["generation"]["missing_information_per_family"] = 1

    class SourceIdRunner(FakeRunner):
        async def derive_missing(self, *args, **kwargs):
            extra = await super().derive_missing(*args, **kwargs)
            for citation in extra.citations:
                citation.source_file = "fixture"
                citation.official_source_url = "https://incorrect.example/writer-value"
            return extra

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=SourceIdRunner)
    assert result["status"] == "complete"
    citations = json.loads((output / "citations.json").read_text())
    assert all(row["source_file"] == "fixture.html" for row in citations)
    assert all(row["official_source_url"] == "https://official.example/fixture" for row in citations)


class TemporalTemplateRunner(FakeRunner):
    """Save minimal planning/checkpoint evidence around an intentional year leak."""

    persistent_mismatch = False

    def __init__(self, *args):
        super().__init__(*args)
        self.work = args[3]
        self.attempts = {}

    async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
        family_id = assignment["family_id"]
        self.attempts[family_id] = self.attempts.get(family_id, 0) + 1
        attempt = self.attempts[family_id]
        draft = await super().create(expert_id, assignment, previous_topics, feedback, previous, replan)
        parent, child = draft.questions[:2]
        parent.note = "Synthetic temporal template test."
        mismatch = family_id == "A-F01" and (attempt == 1 or self.persistent_mismatch)
        if family_id == "A-F01" and attempt > 1:
            assert any(issue.startswith(f"Question {parent.no}:") for issue in feedback)
            assert any(issue.startswith(f"Question {child.no}:") for issue in feedback)
            assert previous is not None and previous.questions[1].question != parent.question
            assert not any(event[0] == "review" and event[2] == family_id for event in self.events)
            assert json.loads((self.work.parent / "questions.json").read_text()) == []
        template = f"NOT LAW fixture {family_id} alpha, input 2 USD in {{case_year}}; payment on 31 December {{case_year}}."
        if not mismatch and family_id == "A-F01":
            # A correction may rewrite both questions together, not just the counterpart.
            template = f"NOT LAW fixture {family_id} alpha, input 2 USD for {{case_year}}; payment at that year's end."
        parent.question = template.replace("{case_year}", str(parent.case_year))
        child_template = (template.replace("31 December {case_year}", f"31 December {parent.case_year}")
                          if mismatch else template)
        child.question = child_template.replace("{case_year}", str(child.case_year))
        cases = [{"question_no": row.no, "variant": row.variant, "tax_category": row.tax_category,
                  "derived_from": parent.no if row.no == child.no else None} for row in draft.questions]
        pipeline.save_json(self.work / "plans" / f"{family_id}.json", {
            "family_id": family_id, "topic": draft.topic, "cases": cases,
            "category_reallocation_reason": None})
        for row, saved_template in ((parent, template), (child, child_template)):
            pipeline.save_json(self.work / "questions" / family_id / f"{row.no}.json", {
                "question_template": saved_template})
        return draft

    async def review(self, reviewer_id, draft, summaries):
        parent, child = draft.questions[:2]
        saved = json.loads((self.work / "questions" / parent.family / f"{parent.no}.json").read_text())
        assert child.question == saved["question_template"].replace("{case_year}", str(child.case_year))
        return await super().review(reviewer_id, draft, summaries)


async def test_temporal_year_leak_repairs_both_cases_before_review_then_resumes_without_calls(setup):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=TemporalTemplateRunner)
    assert result["status"] == "complete"
    assert result["validated_questions"] == 10
    assert [(event[0], event[2]) for event in FakeRunner.events] == [
        ("create", "A-F01"), ("create", "A-F01"), ("review", "A-F01"),
        ("create", "A-F02"), ("review", "A-F02")]
    exported = json.loads((output / "questions.json").read_text())
    assert "payment at that year's end" in exported[0]["question"]
    assert "payment at that year's end" in exported[1]["question"]
    before = list(FakeRunner.events)
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=TemporalTemplateRunner)
    assert resumed["status"] == "complete" and FakeRunner.events == before


async def test_unchanged_temporal_mismatch_stays_pending_and_next_family_can_complete(setup):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2

    class UnchangedTemporalRunner(TemporalTemplateRunner):
        persistent_mismatch = True

    result = await pipeline.run_pipeline(config, sources, output, runner_factory=UnchangedTemporalRunner)
    assert result["status"] == "partial" and result["pause_reason"] is None
    assert result["validated_questions"] == 5
    assert [(event[0], event[2]) for event in FakeRunner.events] == [
        ("create", "A-F01"), ("create", "A-F01"), ("create", "A-F02"), ("review", "A-F02")]
    assert {row["family"] for row in json.loads((output / "questions.json").read_text())} == {"A-F02"}
    assert result["unresolved"][0]["family_id"] == "A-F01"
    assert result["unresolved"][0]["status"] == "needs_correction"
    assert any(issue.startswith("Question 1:") for issue in result["unresolved"][0]["issues"])
    assert any(issue.startswith("Question 2:") for issue in result["unresolved"][0]["issues"])


class MissingPlaceholderRunner(FakeRunner):
    missing_index = 0
    persistent_placeholder_error = False

    def __init__(self, *args):
        super().__init__(*args)
        self.work = args[3]
        self.attempts = {}

    async def create(self, expert_id, assignment, previous_topics, feedback=None, previous=None, replan=False):
        family_id = assignment["family_id"]
        self.attempts[family_id] = self.attempts.get(family_id, 0) + 1
        attempt = self.attempts[family_id]
        draft = await super().create(expert_id, assignment, previous_topics, feedback, previous, replan)
        draft.questions[0].note = "Synthetic placeholder fixture."
        malformed = family_id == "A-F01" and (attempt == 1 or self.persistent_placeholder_error)
        target = draft.questions[self.missing_index]
        if family_id == "A-F01" and attempt > 1:
            assert any(issue.startswith(f"Question {target.no}:") and "{case_year}" in issue for issue in feedback)
            assert previous is not None and previous.questions == draft.questions
            assert not any(event[0] == "review" and event[2] == family_id for event in self.events)
            assert json.loads((self.work.parent / "questions.json").read_text()) == []
        cases = [{"question_no": row.no, "variant": row.variant, "tax_category": row.tax_category,
                  "derived_from": draft.questions[0].no if row.variant == "temporal" else None}
                 for row in draft.questions]
        pipeline.save_json(self.work / "plans" / f"{family_id}.json", {
            "family_id": family_id, "topic": draft.topic, "cases": cases,
            "category_reallocation_reason": None})
        for index, row in enumerate(draft.questions):
            template = row.question if malformed and index == self.missing_index else row.question.replace(str(row.case_year), "{case_year}")
            pipeline.save_json(self.work / "questions" / family_id / f"{row.no}.json", {
                "question_template": template})
        return draft

    async def review(self, reviewer_id, draft, summaries):
        for row in draft.questions:
            saved = json.loads((self.work / "questions" / row.family / f"{row.no}.json").read_text())
            assert "{case_year}" in saved["question_template"]
            assert row.question == saved["question_template"].replace("{case_year}", str(row.case_year))
        return await super().review(reviewer_id, draft, summaries)


@pytest.mark.parametrize("missing_index", [0, 1], ids=["original", "temporal"])
async def test_missing_placeholder_is_corrected_before_review_without_changing_public_facts(setup, missing_index):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2

    class CorrectingRunner(MissingPlaceholderRunner):
        pass

    CorrectingRunner.missing_index = missing_index
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=CorrectingRunner)
    assert result["status"] == "complete"
    assert result["validated_questions"] == 10
    assert [(event[0], event[2]) for event in FakeRunner.events] == [
        ("create", "A-F01"), ("create", "A-F01"), ("review", "A-F01"),
        ("create", "A-F02"), ("review", "A-F02")]
    before = list(FakeRunner.events)
    resumed = await pipeline.run_pipeline(config, sources, output, runner_factory=CorrectingRunner)
    assert resumed["status"] == "complete" and FakeRunner.events == before


@pytest.mark.parametrize("missing_index", [0, 1], ids=["original", "temporal"])
async def test_persistent_missing_placeholder_stays_pending_and_allows_later_family(setup, missing_index):
    config, sources, output, _ = setup
    config["generation"]["families"] = 2

    class UnchangedRunner(MissingPlaceholderRunner):
        persistent_placeholder_error = True

    UnchangedRunner.missing_index = missing_index
    result = await pipeline.run_pipeline(config, sources, output, runner_factory=UnchangedRunner)
    assert result["status"] == "partial" and result["pause_reason"] is None
    assert result["validated_questions"] == 5
    assert [(event[0], event[2]) for event in FakeRunner.events] == [
        ("create", "A-F01"), ("create", "A-F01"), ("create", "A-F02"), ("review", "A-F02")]
    assert {row["family"] for row in json.loads((output / "questions.json").read_text())} == {"A-F02"}
    pending = result["unresolved"][0]
    assert pending["family_id"] == "A-F01" and pending["status"] == "needs_correction"
    assert any(issue.startswith(f"Question {missing_index + 1}:") and "{case_year}" in issue for issue in pending["issues"])

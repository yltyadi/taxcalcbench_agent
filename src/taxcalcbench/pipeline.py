"""Official sources, planning, regular writing and validation, then optional derivation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import __version__
from .schema import FamilyDraft, ReviewReport, normalize_and_check_family
from .settings import allocate_counts, build_assignments, category_options
from .sources import download_sources, policy_extension
from .tools import calculate

ROOT = Path(__file__).resolve().parents[2]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def review_failures(draft: FamilyDraft, report: ReviewReport, author: str) -> list[str]:
    """Check peer approval and recompute the reviewer's explicit final arithmetic."""
    rows = draft.model_dump(by_alias=True)["questions"]
    review = report.model_dump()
    errors = list(review["family_issues"])
    if review.get("plan_revision_needed"):
        errors.append("Reviewer requests a revised family plan")
    if review["reviewer"] != ({"A": "B", "B": "A"}[author]):
        errors.append("Only the other expert may approve this family")
    if review["family_id"] != rows[0]["family"]:
        errors.append("Review refers to a different family")
    results = review["questions"]
    numbers = [r["question_no"] for r in results]
    if sorted(numbers) != sorted(row["no"] for row in rows):
        errors.append("Reviewer must check each question exactly once")
    by_number = {r["question_no"]: r for r in results}
    for row in rows:
        item = by_number.get(row["no"])
        if item is None:
            continue
        if not item["passed"] or item["issues"]:
            errors.append(f"Question {row['no']}: " + "; ".join(item["issues"] or ["review failed"]))
            continue
        try:
            computed = Decimal(calculate(item["calculation"] or ""))
            if computed != Decimal(item["recomputed_answer"]) or computed != Decimal(row["answer_value"]):
                errors.append(f"Question {row['no']}: reviewer calculation and answer disagree")
        except (ValueError, InvalidOperation, TypeError, ArithmeticError):
            errors.append(f"Question {row['no']}: missing or invalid reviewer calculation")
        if (item["unit"] or "").strip().casefold() != row["unit"].strip().casefold():
            errors.append(f"Question {row['no']}: reviewer unit differs from the answer unit")
    return errors


def coverage(rows: list[dict], config: dict, target: int, *, require_full: bool = True) -> dict:
    """Report regular-case period balance overall and within each actual category."""
    generation = config["generation"]
    regular = [row for row in rows if row["variant"] != "missing_information"]
    missing = [row for row in rows if row["variant"] == "missing_information"]
    periods = generation["periods"]
    weights = [period.get("weight", 1) for period in periods]
    expected = allocate_counts(target * generation.get("regular_questions_per_family", 5), weights)
    counts = {period["label"]: sum(period["start"] <= row["case_year"] <= period["end"] for row in regular)
              for period in periods}
    categories = Counter(row["tax_category"] for row in regular if row["variant"] == "original")
    by_category, issues = {}, []
    for category in sorted({row["tax_category"] for row in regular}):
        own = [row for row in regular if row["tax_category"] == category]
        desired = allocate_counts(len(own), weights)
        actual = {period["label"]: sum(period["start"] <= row["case_year"] <= period["end"] for row in own)
                  for period in periods}
        by_category[category] = {"regular_questions": len(own), "periods": actual,
            "period_targets": dict(zip(counts, desired))}
        if len(regular) == target * generation.get("regular_questions_per_family", 5):
            for period, number in zip(periods, desired):
                if abs(actual[period["label"]] - number) > 1:
                    issues.append(f"{category}: period {period['label']} is outside rounding tolerance")
    for family_id in sorted({row["family"] for row in regular}):
        own = [row for row in regular if row["family"] == family_id]
        represented = {row["year_period_target"] for row in own}
        if len(represented) < generation.get("min_periods_per_family", 2):
            issues.append(f"{family_id}: insufficient temporal diversity")
    if len(regular) == target * generation.get("regular_questions_per_family", 5):
        for period, number in zip(periods, expected):
            if counts[period["label"]] != number:
                issues.append(f"{period['label']}: need {number} regular cases")
    return {"A": {"questions": len(rows), "regular_questions": len(regular),
        "missing_information_questions": len(missing), "categories": dict(categories),
        "periods": counts, "period_targets": dict(zip(counts, expected)), "by_category": by_category,
        "coverage_required": True, "issues": issues}}


def _dataset_identity(signature: dict) -> dict:
    config = signature['configuration']
    generation = config['generation']
    # Evidence is identified by the checked source hashes, not its machine's
    # filesystem location. Keep source_dir in run.json for diagnostics only.
    return {key: signature.get(key) for key in ('families', 'single_family', 'references')} | {
        'country': config['country'], 'language': config['language'], 'currency': config.get('currency'),
        'generation': {key: generation.get(key) for key in (
            'regular_questions_per_family', 'missing_information_per_family',
            'min_periods_per_family', 'categories', 'periods')}}


def _draft_fingerprint(draft: FamilyDraft | None) -> str:
    if draft is None:
        return ''
    value = draft.model_dump(mode='json')
    for row in value['questions']:
        row.pop('status', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


async def run_pipeline(config: dict, source_dir: Path, output: Path, *, families: int | None = None,
                       single_family: bool = False, max_documents: int | None = None,
                       limits: dict | None = None, runner_factory=None) -> dict:
    """Prepare a fixed corpus, then finish each family before starting the next."""
    from .experts import (
        BudgetExceeded,
        ExpertRunError,
        ExpertRunner,
        ProviderUnavailable,
        _SourceGap,
        _SourceIntegrityError,
    )

    if single_family and families is not None:
        raise ValueError('Choose either single-family or families')
    if config['generation'].get('blocked_reason'):
        raise ValueError('Generation blocked: ' + config['generation']['blocked_reason'])
    target = 1 if single_family else (families if families is not None else config['generation']['families'])
    if type(target) is not int or target < 1:
        raise ValueError('families must be a positive integer')
    assignments = build_assignments(config, target, creators=('A',))['A']
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    work = output / 'work'
    progress('Preparing official local sources')
    inventory = await asyncio.to_thread(download_sources, config, source_dir, max_documents=max_documents)
    if inventory['country'] != config['country'] or inventory['language'] != config['language']:
        raise ValueError('Source folder belongs to a different country or language')
    if not any(doc.get('kind') != 'navigation' for doc in inventory['documents']):
        raise ValueError('No readable official laws; inspect source download failures')
    progress(f"Using {len(inventory['documents'])} local source documents; sources are fixed for this run")
    signature = {
        'version': __version__, 'configuration': config, 'families': target, 'single_family': single_family,
        'source_dir': str(source_dir),
        'sources': [{key: doc[key] for key in ('id', 'sha256', 'text_sha256')} for doc in inventory['documents']],
        'code': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(Path(__file__).parent.glob('*.py'))},
        'prompts': {path.name: path.read_text() for path in sorted((ROOT / 'prompts').glob('*.md'))},
        'references': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((ROOT / 'references').iterdir()) if path.is_file()},
    }
    if (work / 'run.json').exists():
        before = read_json(work / 'run.json')
        if _dataset_identity(before) != _dataset_identity(signature):
            raise ValueError('Dataset settings changed; use a new output folder')
        old = {doc['id']: doc for doc in before['sources']}
        current = {doc['id']: doc for doc in signature['sources']}
        if any(current.get(key) != value for key, value in old.items()):
            raise ValueError('Existing source documents changed or were removed; use a new output folder')
        previous_policy = before['configuration'].get('sources', {})
        if previous_policy != config.get('sources', {}) and not policy_extension(previous_policy, config.get('sources', {})):
            raise ValueError('Source policy change is not additive; use a new output folder')
        if before != signature:
            revisions_path = work / 'revisions.json'
            revisions = read_json(revisions_path) if revisions_path.exists() else []
            revisions.append({'reason': 'Run settings/code updated; saved evidence and approvals retained',
                              'previous': before})
            save_json(revisions_path, revisions)
            progress('Resuming saved work; approved questions are retained')
    elif any(output.glob('*.json')) or (work.exists() and any(work.iterdir())):
        raise ValueError('Output contains unrelated files; use an empty output folder')
    save_json(work / 'run.json', signature)
    states = {path.stem: read_json(path) for path in sorted((work / 'families').glob('*.json'))}
    runner = (runner_factory or ExpertRunner)(config, inventory, source_dir, work, limits or {})
    paused, pause_reason = False, None

    def save_state(state):
        save_json(work / 'families' / f"{state['assignment']['family_id']}.json", state)

    def summaries():
        return [{'family_id': key, 'topic': state['draft']['topic']}
                for key, state in sorted(states.items()) if state.get('draft')]

    def normalize(draft, assignment):
        draft, errors = normalize_and_check_family(draft, config, inventory, source_dir,
            family_id=assignment['family_id'], question_numbers=assignment['regular_question_numbers'])
        slots = {slot['question_no']: slot for slot in assignment['period_slots']}
        if {row.no for row in draft.questions} != set(slots) or any(
                row.variant == 'missing_information' for row in draft.questions):
            errors.append('Regular creation/review must contain exactly the assigned regular cases')
        for row in draft.questions:
            slot = slots.get(row.no)
            if slot and (row.year_period_target != slot['label'] or not slot['start'] <= row.case_year <= slot['end']):
                errors.append(f'Question {row.no}: case year differs from its assigned temporal slot')
            checkpoint = work / 'questions' / assignment['family_id'] / f'{row.no}.json'
            if checkpoint.exists():
                template = read_json(checkpoint).get('question_template')
                if not template or '{case_year}' not in template:
                    errors.append(f'Question {row.no}: use the literal {{case_year}} placeholder for the scenario year '
                                  'in the question template, preserving all amounts and fixed historical dates. '
                                  'Keep the answer and citations unless another correction requires changes.')
                if template and row.question != template.replace('{case_year}', str(row.case_year)):
                    errors.append(f'Question {row.no}: text differs from its saved year template')
        plan_path = work / 'plans' / f"{assignment['family_id']}.json"
        if plan_path.exists():
            rows = {row.no: row for row in draft.questions}
            for case in read_json(plan_path)['cases']:
                if case['variant'] != 'temporal':
                    continue
                parent_no, number = case['derived_from'], case['question_no']
                parent_path = work / 'questions' / assignment['family_id'] / f'{parent_no}.json'
                template = read_json(parent_path).get('question_template') if parent_path.exists() else None
                if not template or number not in rows:
                    errors.append(f'Question {number}: missing shared temporal question/template')
                elif rows[number].question != template.replace('{case_year}', str(rows[number].case_year)):
                    errors.extend([
                        f'Question {parent_no}: temporal counterpart {number} changed the shared wording or facts. '
                        'Revise the shared scenario to be legally possible in both years. Keep material inputs; '
                        'describe changing statutory conditions generically when possible (for example, statutory '
                        'deadlines rather than one year\'s calendar dates). Recompute the answer if needed.',
                        f'Question {number}: temporal wording or facts differ from Question {parent_no}. '
                        'Use its corrected required_question_template exactly and independently solve for your case year. '
                        'Do not hide a conflicting legal condition by copying an incompatible scenario.',
                    ])
        categories = {row.tax_category for row in draft.questions}
        enabled = {c['label'] for c in config['generation']['categories'] if c.get('enabled', True)}
        if len(categories) != 1 or not categories <= enabled:
            errors.append('A family must use one enabled OECD category')
        if not categories <= set(assignment.get('category_options', enabled)):
            errors.append('Category reallocation would break planned temporal balance')
        if categories != {assignment['category']}:
            path = work / 'plans' / f"{assignment['family_id']}.json"
            if not path.exists() or not read_json(path).get('category_reallocation_reason'):
                errors.append('Changing the target category requires a recorded planning reason')
        return draft, errors

    def export():
        questions, citations = [], []
        seen = set()
        for key, state in sorted(states.items()):
            if state['status'] != 'validated':
                continue
            draft, errors = normalize(FamilyDraft.model_validate(state['draft']), state['assignment'])
            if state.get('review'):
                errors += review_failures(draft, ReviewReport.model_validate(state['review']), 'A')
            else:
                errors.append('A saved approval requires its independent review')
            texts = [' '.join(row.question.split()).casefold() for row in draft.questions]
            if len(set(texts)) != len(texts) or any(text in seen for text in texts):
                errors.append('Duplicate question text')
            if errors:
                state.update(status='needs_correction', issues=errors)
                save_state(state)
                continue
            rows = draft.model_dump(mode='json')
            for row in rows['questions']:
                row['status'] = 'validated'
            extra = state.get('missing_draft')
            if extra:
                combined = FamilyDraft(topic=draft.topic, questions=[*draft.questions, *extra['questions']],
                                       citations=[*draft.citations, *extra['citations']])
                combined, errors = normalize_and_check_family(combined, config, inventory, source_dir,
                    family_id=key, question_numbers=state['assignment']['question_numbers'])
                extra_texts = [' '.join(row.question.split()).casefold() for row in combined.questions
                               if row.variant == 'missing_information']
                if len(set(extra_texts)) != len(extra_texts) or any(text in seen or text in texts for text in extra_texts):
                    errors.append('Duplicate missing-information question text')
                if errors:
                    state.update(missing_draft=None, missing_issues=errors)
                    save_state(state)
                else:
                    numbers = set(state['assignment']['missing_question_numbers'])
                    bound = combined.model_dump(mode='json')
                    rows['questions'].extend(row for row in bound['questions'] if row['no'] in numbers)
                    rows['citations'].extend(row for row in bound['citations'] if row['question_no'] in numbers)
                    seen.update(extra_texts)
            seen.update(texts)
            questions.extend(rows['questions'])
            citations.extend(rows['citations'])
        questions.sort(key=lambda row: row['no'])
        citations.sort(key=lambda row: (row['question_no'], int(row['citation_id'][1:])))
        save_json(output / 'questions.json', questions)
        save_json(output / 'citations.json', citations)
        return questions, citations

    async def finish_regular(state):
        assignment = state['assignment']
        family_id = assignment['family_id']
        feedback = state.pop('resume_issues', state.get('issues', []))
        draft = FamilyDraft.model_validate(state['draft']) if state.get('draft') else None
        needs_write = draft is None or bool(feedback) or state.get('replan', False)
        replan = state.get('replan', False) or bool(feedback and (state.get('review') or {}).get('plan_revision_needed'))
        failed_drafts, source_gaps = set(), set()
        while True:
            try:
                if needs_write:
                    progress(f'{family_id}: planning/writing' if draft is None else f'{family_id}: applying saved feedback')
                    state.update(status='correcting' if draft else 'pending', issues=feedback, replan=replan)
                    save_state(state)
                    draft = await runner.create('A', assignment, summaries(), feedback=feedback or None,
                                                previous=draft, replan=replan)
                    replan = False
                draft, errors = normalize(draft, assignment)
                state.update(draft=draft.model_dump(mode='json'), status='created', issues=errors, replan=False)
                save_state(state)
                # A repaired internal template can clear local errors without
                # changing the rendered question or its answer.
                fingerprint = (_draft_fingerprint(draft), tuple(errors))
                if fingerprint in failed_drafts:
                    state.update(status='needs_correction', issues=feedback or errors)
                    progress(f'{family_id}: unchanged rejected draft; saved for resume')
                    save_state(state)
                    return
                if not errors:
                    progress(f'{family_id}: independent review')
                    report = await runner.review('B', draft, summaries())
                    errors = review_failures(draft, report, 'A')
                    state['review'] = report.model_dump(mode='json')
                    replan = report.plan_revision_needed
                if not errors:
                    state.update(status='validated', issues=[], replan=False)
                    save_state(state)
                    progress(f'{family_id}: {len(draft.questions)} regular questions validated')
                    return
                failed_drafts.add(fingerprint)
                feedback, needs_write = errors, True
                state.update(status='needs_correction', issues=feedback, replan=replan)
                save_state(state)
            except _SourceGap as error:
                detail = str(error)
                state.update(status='needs_correction', issues=[detail], replan=True)
                save_state(state)
                if error.stage == 'plan' or detail in source_gaps:
                    progress(f'{family_id}: no supported revision in this fixed corpus; gap saved')
                    return
                source_gaps.add(detail)
                feedback, replan, needs_write = [detail], True, True
                progress(f'{family_id}: replanning with the available official laws')

    try:
        export()  # Recover already approved rows before any model request.
        for assignment in assignments:
            family_id = assignment['family_id']
            state = states.setdefault(family_id, {'expert': 'A', 'assignment': assignment,
                'status': 'pending', 'draft': None, 'review': None, 'issues': []})
            # The initial export already checked these approvals and extras.
            # Skip finished work instead of re-exporting the entire dataset twice per family.
            if state['status'] == 'validated' and (not assignment['missing_question_numbers'] or state.get('missing_draft')):
                continue
            selected = {path.stem: read_json(path)['cases'][0]['tax_category']
                        for path in sorted((work / 'plans').glob('*.json'))}
            if state['status'] != 'validated':
                state['assignment']['category_options'] = category_options(config, assignments, selected, family_id)
            try:
                if state['status'] != 'validated':
                    await finish_regular(state)
                export()  # Approved regular work survives failure in optional derivation.
                if state['status'] == 'validated' and assignment['missing_question_numbers'] and not state.get('missing_draft'):
                    progress(f'{family_id}: deriving missing-information questions')
                    regular = FamilyDraft.model_validate(state['draft'])
                    for row in regular.questions:
                        row.status = 'validated'
                    extra = await runner.derive_missing('A', assignment, regular)
                    if extra is None or {row.no for row in extra.questions} != set(assignment['missing_question_numbers']) or any(
                            row.variant != 'missing_information' or row.status != 'ready' for row in extra.questions):
                        raise ExpertRunError('Missing-information output differs from the assignment')
                    stored = FamilyDraft.model_validate(state['draft'])
                    combined = FamilyDraft(topic=regular.topic, questions=[*stored.questions, *extra.questions],
                                           citations=[*regular.citations, *extra.citations])
                    _, errors = normalize_and_check_family(combined, config, inventory, source_dir,
                        family_id=family_id, question_numbers=assignment['question_numbers'])
                    if errors:
                        raise ExpertRunError('; '.join(errors))
                    state.update(missing_draft=extra.model_dump(mode='json'), missing_issues=[])
                    save_state(state)
                export()
            except (ProviderUnavailable, BudgetExceeded, ExpertRunError) as error:
                detail = str(error)
                if state['status'] == 'validated':
                    state['missing_issues'] = [detail]
                else:
                    state.update(resume_issues=list(state.get('issues', [])), status='error', issues=[detail])
                save_state(state)
                if state['status'] == 'validated' and not isinstance(error, (ProviderUnavailable, BudgetExceeded, _SourceIntegrityError)):
                    progress(f'{family_id}: optional missing-information question pending: {detail}. Continuing with the next family.')
                    continue
                paused, pause_reason = True, detail
                progress(f'{family_id}: {detail}. Progress saved; rerun the same command to resume.')
                break
        questions, citations = export()
        counts = coverage(questions, config, target)
        regular_count = config['generation'].get('regular_questions_per_family', 5)
        missing_count = config['generation'].get('missing_information_per_family', 1)
        requested = target * (regular_count + missing_count)
        unresolved = [{'family_id': key, 'status': state['status'], 'issues': state.get('issues', [])}
                      for key, state in sorted(states.items()) if state['status'] != 'validated']
        unresolved += [{'family_id': key, 'status': 'missing_information_error', 'issues': state['missing_issues']}
                       for key, state in sorted(states.items()) if state.get('missing_issues')]
        reallocations = [{'family_id': path.stem, 'chosen_category': plan['cases'][0]['tax_category'],
                          'reason': plan['category_reallocation_reason']}
                         for path in sorted((work / 'plans').glob('*.json'))
                         if (plan := read_json(path)).get('category_reallocation_reason')]
        complete = len(questions) == requested and not unresolved and all(not item['issues'] for item in counts.values())
        result = {'status': 'complete' if complete else 'paused' if paused else 'partial', 'pause_reason': pause_reason,
            'country': config['country'], 'language': config['language'], 'single_family': single_family,
            'requested_questions': requested, 'requested_regular_questions': target * regular_count,
            'requested_missing_information_questions': target * missing_count,
            'created_questions': sum(len((s.get('draft') or {}).get('questions', [])) + len((s.get('missing_draft') or {}).get('questions', [])) for s in states.values()),
            'exported_questions': len(questions), 'validated_questions': sum(q['variant'] != 'missing_information' for q in questions),
            'generated_missing_information_questions': sum(q['variant'] == 'missing_information' for q in questions),
            'validated_families': sum(s['status'] == 'validated' for s in states.values()),
            'category_reallocations': reallocations,
            'absent_categories': [c['label'] for c in config['generation']['categories']
                                  if c.get('enabled', True) and c['label'] not in {q['tax_category'] for q in questions}],
            'coverage': counts, 'unresolved': unresolved, 'source_failures': inventory.get('failures', []), 'usage': runner.usage()}
        save_json(work / 'result.json', result)
        return result
    finally:
        close = getattr(runner, 'close', None)
        if close:
            await close()

"""Dataset loading and category-aware family partitioning."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def select_category_balanced_dev_families(
    questions: list[dict[str, Any]], target_count: int = 5
) -> set[str]:
    """Select a balanced set of development families ensuring representation across OECD categories.

    Picks 1 family per distinct tax category present in the dataset, then fills up to
    target_count in a deterministic round-robin manner.
    """
    cat_to_fams: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        fam = q.get("family", "")
        cat = q.get("tax_category", "")
        if fam and cat and fam not in cat_to_fams[cat]:
            cat_to_fams[cat].append(fam)

    sorted_cats = sorted(cat_to_fams.keys())
    selected: set[str] = set()

    # Phase 1: 1 family per distinct category
    for cat in sorted_cats:
        for fam in cat_to_fams[cat]:
            if fam not in selected:
                selected.add(fam)
                break
        if len(selected) >= target_count:
            break

    # Phase 2: If we still need more to reach target_count, round-robin
    while len(selected) < target_count:
        added_any = False
        for cat in sorted_cats:
            for fam in cat_to_fams[cat]:
                if fam not in selected:
                    selected.add(fam)
                    added_any = True
                    break
            if len(selected) >= target_count:
                break
        if not added_any:
            break

    return selected


def load_evaluation_dataset(
    dataset_dir: Path,
    split: str = "full",
    families: list[str] | None = None,
    limit: int | None = None,
    dev_target_count: int = 5,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], set[str]]:
    """Load questions and citations from the validated outputs directory.

    Args:
        dataset_dir: Path containing questions.json and citations.json.
        split: 'full' / 'all' (all 50 families), 'dev' (category-balanced families),
               or 'test' (remaining families).
        families: Optional explicit list of family IDs to filter.
        limit: Optional maximum number of questions to load.
        dev_target_count: Target number of families for the dev split.

    Returns:
        A tuple of (questions_list, citations_map_by_question_no, dev_families_set).
    """
    questions_file = dataset_dir / "questions.json"
    citations_file = dataset_dir / "citations.json"

    if not questions_file.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_file}")
    if not citations_file.exists():
        raise FileNotFoundError(f"Missing citations file: {citations_file}")

    with open(questions_file, encoding="utf-8") as f:
        all_questions = json.load(f)

    with open(citations_file, encoding="utf-8") as f:
        all_citations = json.load(f)

    citations_map: dict[int, list[dict[str, Any]]] = {}
    for citation in all_citations:
        q_no = citation.get("question_no")
        if q_no is not None:
            citations_map.setdefault(q_no, []).append(citation)

    # Compute category-balanced dev families
    dev_families = select_category_balanced_dev_families(all_questions, target_count=dev_target_count)

    split_lower = split.lower().strip()

    if families:
        fam_set = set(families)
        filtered_questions = [q for q in all_questions if q.get("family") in fam_set]
    elif split_lower in ("full", "all"):
        filtered_questions = all_questions
    elif split_lower == "dev":
        filtered_questions = [q for q in all_questions if q.get("family") in dev_families]
    elif split_lower == "test":
        filtered_questions = [q for q in all_questions if q.get("family") not in dev_families]
    else:
        raise ValueError(
            f"Unknown split: '{split}'. Choose 'full' / 'all' (all 300 questions), 'dev', or 'test'."
        )

    if limit and limit > 0:
        filtered_questions = filtered_questions[:limit]

    return filtered_questions, citations_map, dev_families

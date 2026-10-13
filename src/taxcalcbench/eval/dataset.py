"""Dataset loading and family-preserving partitioning."""

import json
from pathlib import Path
from typing import Any

from .config import DEV_FAMILIES


def load_evaluation_dataset(
    dataset_dir: Path, split: str = "dev"
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Load questions and citations from the validated outputs directory.

    Args:
        dataset_dir: Path containing questions.json and citations.json.
        split: 'dev' (5 families), 'test' (45 families), or 'all' (50 families).

    Returns:
        A tuple of (questions_list, citations_map_by_question_no).
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

    if split == "dev":
        filtered_questions = [q for q in all_questions if q.get("family") in DEV_FAMILIES]
    elif split == "test":
        filtered_questions = [q for q in all_questions if q.get("family") not in DEV_FAMILIES]
    elif split == "all":
        filtered_questions = all_questions
    else:
        raise ValueError(f"Unknown split: {split}. Choose 'dev', 'test', or 'all'.")

    return filtered_questions, citations_map

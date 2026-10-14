"""Configuration schemas for TaxCalcBench evaluations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ConditionType = Literal[
    "closed_book_cot",
    "oracle_cot",
    "closed_book_program_cot",
    "oracle_program_cot",
    "bm25_rag_cot",
    "bm25_rag_program_cot",
]


@dataclass
class EvalConfig:
    """Settings for an evaluation run."""

    country: str = "kz"
    language: str = "ru"
    dataset_dir: Path = Path("outputs/full/kz")
    source_dir: Path = Path("data/kz/sources")
    output_dir: Path = Path("outputs/eval")
    model: str = "gemini-3.8-flash"
    conditions: list[ConditionType] = field(default_factory=lambda: ["closed_book_cot", "oracle_cot"])
    split: str = "full"
    families: list[str] | None = None
    limit: int | None = None
    top_k: int = 5
    max_concurrency: int = 15
    timeout_seconds: float = 90.0
    max_retries: int = 5
    temperature: float = 0.0
    max_output_tokens: int = 16384
    reasoning_effort: str = "low"

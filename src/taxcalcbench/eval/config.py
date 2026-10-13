"""Configuration schemas for TaxCalcBench evaluations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ConditionType = Literal[
    "closed_book_cot",
    "oracle_cot",
    "closed_book_program_cot",
    "oracle_program_cot",
]


@dataclass
class EvalConfig:
    """Settings for an evaluation run."""

    country: str = "kz"
    language: str = "ru"
    dataset_dir: Path = Path("outputs/full/kz")
    output_dir: Path = Path("outputs/eval")
    model: str = "gemini-3.8-flash"
    conditions: list[ConditionType] = field(default_factory=lambda: ["closed_book_cot", "oracle_cot"])
    split: str = "full"
    families: list[str] | None = None
    limit: int | None = None
    max_concurrency: int = 5
    timeout_seconds: float = 60.0
    max_retries: int = 10
    temperature: float = 0.0
    max_output_tokens: int = 8192

"""Configuration schemas for TaxCalcBench evaluations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEV_FAMILIES: set[str] = {"A-F01", "A-F02", "A-F03", "A-F04", "A-F05"}

ConditionType = Literal[
    "closed_book_cot",
    "oracle_cot",
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
    split: Literal["dev", "test", "all"] = "dev"
    max_concurrency: int = 5
    temperature: float = 0.0
    max_output_tokens: int = 8192

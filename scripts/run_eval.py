#!/usr/bin/env python3
"""CLI runner script for TaxCalcBench benchmark evaluations.

Supports evaluating any country, full dataset or category-balanced dev/test splits,
configurable timeouts, retries, and conditions (Closed-Book CoT, Oracle CoT, Program-Aided CoT).
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxcalcbench.eval.config import EvalConfig
from taxcalcbench.eval.runner import evaluate_dataset


def main():
    parser = argparse.ArgumentParser(description="Run TaxCalcBench experimental evaluation")
    parser.add_argument("--country", default="kz", help="Country code (e.g. kz, kz_kk, cn, pl, id, pk, eg, in)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Path to outputs/full/<country>")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"), help="Path to save eval outputs")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID (default: gemini-3.8-flash)")
    parser.add_argument(
        "--split",
        choices=["full", "all", "dev", "test"],
        default="full",
        help="Evaluation split: 'full' / 'all' (all 300 questions, default), 'dev' (category-balanced families), or 'test' (remaining families)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Specific family IDs to evaluate (e.g. --families A-F01 A-F02 A-F10)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to the first N questions")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["closed_book_cot", "oracle_cot"],
        help="Conditions to run: closed_book_cot, oracle_cot, closed_book_program_cot, oracle_program_cot",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent model calls (default: 5)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-call timeout in seconds (default: 60.0)")
    parser.add_argument("--retries", type=int, default=10, help="Max retry attempts per question (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir or Path(f"outputs/full/{args.country}")

    config = EvalConfig(
        country=args.country,
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        model=args.model,
        conditions=args.conditions,
        split=args.split,
        families=args.families,
        limit=args.limit,
        max_concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        temperature=args.temperature,
    )

    asyncio.run(evaluate_dataset(config))


if __name__ == "__main__":
    main()

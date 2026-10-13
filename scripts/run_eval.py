#!/usr/bin/env python3
"""CLI runner script for TaxCalcBench benchmark evaluations."""

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
    parser.add_argument("--country", default="kz", help="Country code (default: kz)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Path to outputs/full/<country>")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"), help="Path to save eval outputs")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID (default: gemini-3.8-flash)")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="Split (default: dev)")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["closed_book_cot", "oracle_cot"],
        help="Conditions to run (e.g. closed_book_cot oracle_cot)",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent model calls")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir or Path(f"outputs/full/{args.country}")

    config = EvalConfig(
        country=args.country,
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        model=args.model,
        conditions=args.conditions,
        split=args.split,
        max_concurrency=args.concurrency,
    )

    asyncio.run(evaluate_dataset(config))


if __name__ == "__main__":
    main()

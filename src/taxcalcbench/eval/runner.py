"""Asynchronous execution runner for TaxCalcBench benchmark evaluations."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from .config import EvalConfig
from .dataset import load_evaluation_dataset
from .extractors import extract_answer
from .metrics import compute_metrics
from .prompts import build_eval_prompt
from .rag import BM25Retriever, HybridRRFTRetriever, evaluate_retrieval_hit, load_and_chunk_corpus

load_dotenv()


def get_api_client(timeout_seconds: float = 60.0) -> AsyncOpenAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment or .env")
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=timeout_seconds,
        max_retries=0,
    )


async def call_model_with_retry(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    timeout_seconds: float = 90.0,
    max_retries: int = 5,
    reasoning_effort: str = "low",
) -> str:
    """Call model with robust exponential backoff on transient errors and timeouts."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
                reasoning_effort=reasoning_effort,
            )
            content = response.choices[0].message.content or ""
            if not content.strip() and attempt < max_retries - 1:
                await asyncio.sleep(2.0)
                continue
            return content
        except (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            TimeoutError,
            asyncio.TimeoutError,
            APIError,
        ) as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(60.0, 3.0 * (2.0 ** min(attempt, 4)))
            print(f"Provider {type(exc).__name__}: retry {attempt + 1}/{max_retries}; waiting {delay:.1f}s...")
            await asyncio.sleep(delay)
    return ""


async def evaluate_single_question(
    client: AsyncOpenAI,
    question: dict[str, Any],
    citations: list[dict[str, Any]],
    condition: str,
    config: EvalConfig,
    work_dir: Path,
    sem: asyncio.Semaphore,
    retriever: BM25Retriever | HybridRRFTRetriever | None = None,
) -> dict[str, Any]:
    """Evaluate one question and save its checkpoint."""
    q_no = question["no"]
    checkpoint_file = work_dir / f"{q_no}.json"

    # Reuse existing checkpoint if present and valid
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, encoding="utf-8") as f:
                record = json.load(f)
            # If checkpoint has a valid raw_response and not an error, reuse it
            if record.get("raw_response") and not str(record.get("raw_response")).startswith("[EVALUATION ERROR"):
                return record
        except Exception:
            pass

    retrieved_chunks = None
    retrieval_metrics = None
    if "rag" in condition and retriever is not None:
        query = f"{question.get('case_year', '')} {question.get('tax_category', '')} {question.get('question', '')}"
        if isinstance(retriever, HybridRRFTRetriever):
            mode = "dense" if condition.startswith("dense_rag") else "hybrid"
            async with sem:
                retrieved_chunks = await retriever.retrieve_async(
                    client, query, top_k=config.top_k, case_year=question.get("case_year"), mode=mode
                )
        else:
            retrieved_chunks = retriever.retrieve(query, top_k=config.top_k, case_year=question.get("case_year"))
        retrieval_metrics = evaluate_retrieval_hit(retrieved_chunks, citations)

    prompt = build_eval_prompt(question, condition, citations, retrieved_chunks=retrieved_chunks)

    try:
        async with sem:
            t0 = time.perf_counter()
            raw_response = await call_model_with_retry(
                client=client,
                model=config.model,
                prompt=prompt,
                temperature=config.temperature,
                max_tokens=config.max_output_tokens,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                reasoning_effort=config.reasoning_effort,
            )
            elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"Warning: Question {q_no} evaluation failed after retries ({type(exc).__name__}: {exc})")
        raw_response = f"[EVALUATION ERROR: {type(exc).__name__}: {exc}]"
        elapsed = 0.0

    is_program = "program_cot" in condition
    pred_value, pred_unit = extract_answer(raw_response, is_program=is_program)

    record = {
        "question_no": q_no,
        "family": question.get("family"),
        "variant": question.get("variant"),
        "case_year": question.get("case_year"),
        "year_period_target": question.get("year_period_target"),
        "tax_category": question.get("tax_category"),
        "difficulty": question.get("difficulty"),
        "condition": condition,
        "model": config.model,
        "gold_answer": str(question.get("answer_value", "")).strip(),
        "gold_unit": question.get("unit"),
        "pred_value": pred_value,
        "pred_unit": pred_unit,
        "elapsed_seconds": round(elapsed, 2),
        "retrieval_metrics": retrieval_metrics,
        "prompt": prompt,
        "raw_response": raw_response,
    }

    # Save incremental checkpoint
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return record


async def run_evaluation_condition(
    condition: str,
    questions: list[dict[str, Any]],
    citations_map: dict[int, list[dict[str, Any]]],
    config: EvalConfig,
) -> dict[str, Any]:
    """Run evaluation across all questions for a single condition."""
    print(f"\n=======================================================")
    print(f"Running condition: {condition} ({len(questions)} questions)")
    print(f"Model: {config.model} | Split: {config.split}")
    print(f"=======================================================")

    client = get_api_client(timeout_seconds=config.timeout_seconds)
    dir_name = f"{condition}_k{config.top_k}" if ("rag" in condition and config.top_k != 5) else condition
    cond_dir = config.output_dir / config.country / dir_name
    work_dir = cond_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    retriever: BM25Retriever | HybridRRFTRetriever | None = None
    if "rag" in condition:
        print(f"Building statutory RAG index from {config.source_dir}...")
        chunks = load_and_chunk_corpus(config.source_dir)
        use_temporal = any(k in condition for k in ("temporal", "dense", "hybrid"))
        bm25 = BM25Retriever(chunks, source_dir=config.source_dir, temporal_aware=use_temporal)
        if condition.startswith(("dense_rag", "hybrid_rag")):
            cache_dir = config.output_dir / config.country / "rag_index"
            retriever = HybridRRFTRetriever(bm25, cache_dir=cache_dir)
            await retriever.prewarm_questions(client, questions)
        else:
            retriever = bm25
        print(f"Indexed {len(chunks)} statutory chunks (top_k={config.top_k}, temporal_aware={use_temporal})")

    sem = asyncio.Semaphore(config.max_concurrency)

    tasks = [
        evaluate_single_question(
            client=client,
            question=q,
            citations=citations_map.get(q["no"], []),
            condition=condition,
            config=config,
            work_dir=work_dir,
            sem=sem,
            retriever=retriever,
        )
        for q in questions
    ]

    records = await asyncio.gather(*tasks)
    records.sort(key=lambda r: r["question_no"])

    if isinstance(retriever, HybridRRFTRetriever):
        retriever.save_cache()

    # Compute metrics
    metrics = compute_metrics(records)

    # Save summary
    summary = {
        "country": config.country,
        "split": config.split,
        "condition": condition,
        "model": config.model,
        "metrics": metrics,
    }

    summary_file = cond_dir / f"summary_{config.split}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    records_file = cond_dir / f"records_{config.split}.json"
    with open(records_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nCondition {condition} complete!")
    print(f"  Exact Match (All):      {metrics['overall_exact_match'] * 100:.1f}%")
    print(f"  Tolerant Match (All):   {metrics['overall_tolerant_match'] * 100:.1f}%")
    print(f"  Solvable Exact Match:   {metrics['solvable_exact_match'] * 100:.1f}%")
    print(f"  Missing Info Accuracy:  {metrics['abstention_accuracy'] * 100:.1f}%")
    print(f"  Family Consistency:     {metrics['family_consistency_score'] * 100:.1f}%")

    return summary


async def evaluate_dataset(config: EvalConfig) -> dict[str, Any]:
    """Execute evaluation across all configured conditions."""
    questions, citations_map, dev_families = load_evaluation_dataset(
        config.dataset_dir,
        split=config.split,
        families=config.families,
        limit=config.limit,
    )
    print(
        f"Loaded {len(questions)} questions for country '{config.country}' "
        f"(split='{config.split}', dev_families={sorted(dev_families)})"
    )

    results = {}
    for condition in config.conditions:
        summary = await run_evaluation_condition(condition, questions, citations_map, config)
        results[condition] = summary

    # Save comparison summary
    comp_file = config.output_dir / config.country / f"comparison_{config.split}.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

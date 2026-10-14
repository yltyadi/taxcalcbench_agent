"""Unit tests for the TaxCalcBench experimental evaluation framework."""

from pathlib import Path

from taxcalcbench.eval.dataset import select_category_balanced_dev_families
from taxcalcbench.eval.extractors import extract_answer
from taxcalcbench.eval.metrics import compute_metrics, is_exact_match, is_tolerant_match
from taxcalcbench.eval.prompts import build_eval_prompt
from taxcalcbench.eval.rag import BM25Retriever, evaluate_retrieval_hit
from taxcalcbench.eval.rag.chunker import chunk_document_text
from taxcalcbench.eval.sandbox import execute_python_code


def test_category_balanced_dev_selection():
    questions = [
        {"family": "A-F01", "tax_category": "CatA"},
        {"family": "A-F02", "tax_category": "CatA"},
        {"family": "A-F03", "tax_category": "CatB"},
        {"family": "A-F04", "tax_category": "CatC"},
    ]
    selected = select_category_balanced_dev_families(questions, target_count=3)
    assert selected == {"A-F01", "A-F03", "A-F04"}


def test_extract_answer_boxed_and_program():
    val, unit = extract_answer("Reasoning steps...\n[FINAL ANSWER]: \\boxed{500 000} KZT")
    assert val == "500000"
    assert unit == "KZT"

    val_abs, unit_abs = extract_answer("[FINAL ANSWER]: \\boxed{insufficient information}")
    assert val_abs == "insufficient information"
    assert unit_abs is None

    code_resp = "```python\nfrom decimal import Decimal\nans = Decimal('120000') * Decimal('0.10')\n```"
    val_prog, _ = extract_answer(code_resp, is_program=True)
    assert is_exact_match(val_prog, "12000")


def test_sandbox_safety_and_execution():
    res = execute_python_code("from decimal import Decimal\nans = Decimal('2500') + Decimal('500')")
    assert res["success"] is True
    assert res["value"] == "3000"

    res_bad = execute_python_code("import os\nans = 1")
    assert res_bad["success"] is False
    assert "ImportError" in res_bad["error"]


def test_metrics_computation():
    records = [
        {
            "question_no": 1,
            "family": "A-F01",
            "variant": "original",
            "year_period_target": "2020-2026",
            "gold_answer": "1000",
            "pred_value": "1000.00",
        },
        {
            "question_no": 2,
            "family": "A-F01",
            "variant": "missing_information",
            "year_period_target": "2020-2026",
            "gold_answer": "insufficient information",
            "pred_value": "insufficient information",
        },
    ]
    m = compute_metrics(records)
    assert m["overall_exact_match"] == 1.0
    assert m["solvable_exact_match"] == 1.0
    assert m["abstention_accuracy"] == 1.0
    assert m["family_consistency_score"] == 1.0
    assert is_tolerant_match("1005", "1000", tolerance=0.01)


def test_rag_chunker_and_bm25():
    text = "Статья 1. Общие положения\nНалог составляет 10 процентов.\nСтатья 2. Ставки налога\nКорпоративный подоходный налог составляет 20 процентов."
    chunks = chunk_document_text("s_test123", "Test Tax Code", text, max_chars=500)
    assert len(chunks) == 2
    retriever = BM25Retriever(chunks)
    top = retriever.retrieve("Корпоративный подоходный налог 20 процентов", top_k=1)
    assert len(top) == 1
    assert "Статья 2" in top[0]["text"]

    hit = evaluate_retrieval_hit(
        top,
        [{"source_file": "raw/s_test123/law.txt", "supporting_passage": "Корпоративный подоходный налог составляет 20 процентов."}],
    )
    assert hit["passage_recall"] == 1.0
    assert hit["complete_evidence_recall"] == 1.0
    assert hit["source_doc_recall"] == 1.0

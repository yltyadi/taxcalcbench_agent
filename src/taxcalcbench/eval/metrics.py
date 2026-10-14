"""Evaluation metrics calculation for TaxCalcBench benchmark."""

from decimal import Decimal, InvalidOperation
from typing import Any

from .extractors import INSUFFICIENT_INFO


def is_exact_match(pred: str | None, gold: str) -> bool:
    """Check if prediction is a strict exact match to gold answer."""
    if pred is None:
        return False

    pred_clean = pred.strip().lower()
    gold_clean = gold.strip().lower()

    if gold_clean == INSUFFICIENT_INFO:
        return pred_clean == INSUFFICIENT_INFO

    if pred_clean == INSUFFICIENT_INFO:
        return False

    try:
        p_dec = Decimal(pred_clean)
        g_dec = Decimal(gold_clean)
        return abs(p_dec - g_dec) <= Decimal("0.01")
    except (InvalidOperation, ValueError):
        return pred_clean == gold_clean


def is_tolerant_match(pred: str | None, gold: str, tolerance: float = 0.01) -> bool:
    """Check if prediction is within 1% relative tolerance or 1.0 absolute margin."""
    if is_exact_match(pred, gold):
        return True

    if pred is None:
        return False

    pred_clean = pred.strip().lower()
    gold_clean = gold.strip().lower()

    if gold_clean == INSUFFICIENT_INFO or pred_clean == INSUFFICIENT_INFO:
        return False

    try:
        p_val = float(pred_clean)
        g_val = float(gold_clean)
        abs_diff = abs(p_val - g_val)
        if abs_diff <= 1.0:
            return True
        denom = max(abs(g_val), 1.0)
        return (abs_diff / denom) <= tolerance
    except ValueError:
        return False


def compute_metrics(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute benchmark metrics from evaluation records.

    Args:
        eval_records: List of records containing question, prediction, extracted_val, gold.

    Returns:
        Dictionary of aggregated metrics and segment breakdowns.
    """
    total = len(eval_records)
    if total == 0:
        return {"total": 0}

    solvable = [r for r in eval_records if r["gold_answer"] != INSUFFICIENT_INFO]
    missing = [r for r in eval_records if r["gold_answer"] == INSUFFICIENT_INFO]

    em_correct = sum(1 for r in eval_records if is_exact_match(r["pred_value"], r["gold_answer"]))
    tol_correct = sum(1 for r in eval_records if is_tolerant_match(r["pred_value"], r["gold_answer"]))

    solvable_em = sum(1 for r in solvable if is_exact_match(r["pred_value"], r["gold_answer"]))
    solvable_tol = sum(1 for r in solvable if is_tolerant_match(r["pred_value"], r["gold_answer"]))

    missing_correct = sum(1 for r in missing if is_exact_match(r["pred_value"], r["gold_answer"]))
    false_abstained = sum(
        1 for r in solvable if (r["pred_value"] or "").strip().lower() == INSUFFICIENT_INFO
    )

    # Family Consistency (all regular solvable questions in family correct)
    family_groups: dict[str, list[dict[str, Any]]] = {}
    for r in eval_records:
        fam = r.get("family", "unknown")
        family_groups.setdefault(fam, []).append(r)

    consistent_families = 0
    total_families = len(family_groups)
    for fam, items in family_groups.items():
        solvable_items = [it for it in items if it["gold_answer"] != INSUFFICIENT_INFO]
        if solvable_items and all(is_exact_match(it["pred_value"], it["gold_answer"]) for it in solvable_items):
            consistent_families += 1

    # By Period breakdown
    periods: dict[str, list[dict[str, Any]]] = {}
    for r in eval_records:
        period = r.get("year_period_target", "other")
        periods.setdefault(period, []).append(r)

    period_metrics = {}
    for p_name, p_items in sorted(periods.items()):
        p_solvable = [it for it in p_items if it["gold_answer"] != INSUFFICIENT_INFO]
        p_em = sum(1 for it in p_solvable if is_exact_match(it["pred_value"], it["gold_answer"]))
        period_metrics[p_name] = {
            "total": len(p_items),
            "solvable": len(p_solvable),
            "acc_em": round(p_em / len(p_solvable), 4) if p_solvable else 0.0,
        }

    # By Variant breakdown
    variants: dict[str, list[dict[str, Any]]] = {}
    for r in eval_records:
        var = r.get("variant", "other")
        variants.setdefault(var, []).append(r)

    variant_metrics = {}
    for v_name, v_items in sorted(variants.items()):
        v_em = sum(1 for it in v_items if is_exact_match(it["pred_value"], it["gold_answer"]))
        variant_metrics[v_name] = {
            "total": len(v_items),
            "acc_em": round(v_em / len(v_items), 4) if v_items else 0.0,
        }

    # Aggregate retrieval metrics if present
    ret_records = [r["retrieval_metrics"] for r in eval_records if r.get("retrieval_metrics")]
    retrieval_summary = None
    if ret_records:
        n_ret = len(ret_records)
        retrieval_summary = {
            "passage_recall": round(sum(x["passage_recall"] for x in ret_records) / n_ret, 4),
            "complete_evidence_recall": round(sum(x["complete_evidence_recall"] for x in ret_records) / n_ret, 4),
            "source_doc_recall": round(sum(x["source_doc_recall"] for x in ret_records) / n_ret, 4),
            "mrr": round(sum(x["mrr"] for x in ret_records) / n_ret, 4),
        }

    return {
        "total_questions": total,
        "overall_exact_match": round(em_correct / total, 4) if total else 0.0,
        "overall_tolerant_match": round(tol_correct / total, 4) if total else 0.0,
        "solvable_questions": len(solvable),
        "solvable_exact_match": round(solvable_em / len(solvable), 4) if solvable else 0.0,
        "solvable_tolerant_match": round(solvable_tol / len(solvable), 4) if solvable else 0.0,
        "missing_info_questions": len(missing),
        "abstention_accuracy": round(missing_correct / len(missing), 4) if missing else 0.0,
        "false_abstention_rate": round(false_abstained / len(solvable), 4) if solvable else 0.0,
        "total_families": total_families,
        "family_consistency_score": round(consistent_families / total_families, 4) if total_families else 0.0,
        "retrieval": retrieval_summary,
        "by_period": period_metrics,
        "by_variant": variant_metrics,
    }

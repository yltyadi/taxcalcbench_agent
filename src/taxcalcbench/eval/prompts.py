"""Prompt formatters for TaxCalcBench evaluations."""

from typing import Any


def format_oracle_citations(citations: list[dict[str, Any]]) -> str:
    """Format gold statutory citations into prompt context."""
    if not citations:
        return ""

    lines = ["[OFFICIAL LEGAL CITATIONS]:"]
    for c in citations:
        title = c.get("official_title", "")
        art = c.get("article_section", "")
        passage = c.get("supporting_passage", "").strip()
        loc = f"Article/Section: {art}" if art else ""
        header = f"- {title} ({loc}):" if loc else f"- {title}:"
        lines.append(f'{header}\n  "{passage}"')

    return "\n".join(lines)


def format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved RAG chunks into prompt context."""
    if not chunks:
        return ""

    lines = ["[RETRIEVED OFFICIAL STATUTORY PASSAGES]:"]
    for idx, ch in enumerate(chunks, 1):
        title = ch.get("title", "")
        text = ch.get("text", "").strip()
        lines.append(f"[Passage {idx} — {title}]:\n{text}\n")

    return "\n".join(lines)


def build_eval_prompt(
    question: dict[str, Any],
    condition: str,
    citations: list[dict[str, Any]] | None = None,
    retrieved_chunks: list[dict[str, Any]] | None = None,
) -> str:
    """Build standardized user prompt according to condition."""
    country = question.get("country", "")
    case_year = question.get("case_year", "")
    q_text = question.get("question", "").strip()

    parts = []

    if condition in ("oracle_cot", "oracle_program_cot") and citations:
        context_str = format_oracle_citations(citations)
        if context_str:
            parts.append(context_str)
            parts.append("")
    elif "rag" in condition and retrieved_chunks:
        context_str = format_retrieved_chunks(retrieved_chunks)
        if context_str:
            parts.append(context_str)
            parts.append("")

    parts.append(f"[QUESTION]:\n{q_text}")
    parts.append("")

    if "program_cot" in condition:
        parts.append(
            f"[INSTRUCTIONS]:\n"
            f"Solve this tax calculation for {country} in the year {case_year}.\n"
            f"Express your legal reasoning and exact statutory arithmetic as executable Python code.\n"
            f"Wrap your Python code in a ```python ... ``` block.\n"
            f"Your code must assign the final numeric answer to a variable named `ans` (e.g. ans = Decimal('500000') or ans = 500000) or print it on the last line.\n"
            f'If essential numeric facts are missing and the calculation cannot be completed, assign ans = "insufficient information".'
        )
    else:
        parts.append(
            f"[INSTRUCTIONS]:\n"
            f"Solve this tax calculation for {country} in the year {case_year}.\n"
            f"Think step by step, citing applicable legal provisions and performing exact calculations.\n"
            f"Conclude your response with the final answer on the last line using this exact format:\n"
            f"[FINAL ANSWER]: \\boxed{{value}} [UNIT]\n"
            f'(If essential numeric facts are missing and the calculation cannot be completed, write:\n'
            f"[FINAL ANSWER]: \\boxed{{insufficient information}})"
        )

    return "\n".join(parts)

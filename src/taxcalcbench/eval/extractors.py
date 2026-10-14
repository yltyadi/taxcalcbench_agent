"""Answer extraction and normalization for evaluation responses."""

import re
from typing import Any

from .sandbox import execute_python_code

INSUFFICIENT_INFO = "insufficient information"
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.IGNORECASE)
FINAL_ANSWER_PATTERN = re.compile(r"\[FINAL ANSWER\]:\s*([^\n\r]+)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?(?:[0-9]+(?:[\s,][0-9]+)*(?:\.[0-9]+)?|\.[0-9]+)")
PYTHON_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def normalize_numeric_string(raw: str) -> str:
    """Normalize numeric text by removing spaces and standardizing decimals."""
    cleaned = raw.strip()
    # Remove thousands separators (space or comma between digits)
    cleaned = re.sub(r"(?<=\d)[\s,](?=\d)", "", cleaned)
    # Remove trailing period if not decimal
    cleaned = cleaned.rstrip(".")
    return cleaned


def extract_answer(response_text: str, is_program: bool = False) -> tuple[str | None, str | None]:
    """Extract prediction value and optional unit from model completion.

    Returns:
        (extracted_value, extracted_unit)
    """
    if not response_text:
        return None, None

    text = response_text.strip()

    # 0. If program-aided or code block present, attempt execution first
    if is_program or "```" in text:
        code_blocks = PYTHON_BLOCK_PATTERN.findall(text)
        if code_blocks:
            code = code_blocks[-1].strip()
            exec_res = execute_python_code(code)
            if exec_res.get("success") and exec_res.get("value") is not None:
                val = str(exec_res["value"]).strip()
                if "insufficient information" in val.lower() or "недостаточно информации" in val.lower():
                    return INSUFFICIENT_INFO, None
                num_match = NUMBER_PATTERN.search(val)
                if num_match:
                    return normalize_numeric_string(num_match.group(0)), None
                return val, None

    # 1. Check for boxed answer
    boxed_matches = BOXED_PATTERN.findall(text)
    candidate = boxed_matches[-1].strip() if boxed_matches else None

    # 2. Check for [FINAL ANSWER]: prefix
    if not candidate:
        final_matches = FINAL_ANSWER_PATTERN.findall(text)
        if final_matches:
            candidate = final_matches[-1].strip()

    # 3. Fallback: check the last non-empty line
    if not candidate:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            candidate = lines[-1]

    if not candidate:
        return None, None

    # Check for insufficient information (English or Russian)
    lower_cand = candidate.lower()
    if (
        "insufficient information" in lower_cand
        or "недостаточно информации" in lower_cand
        or "информация отсутствует" in lower_cand
        or "невозможно рассчитать" in lower_cand
    ):
        return INSUFFICIENT_INFO, None

    # Try extracting numeric amount and unit
    # Separate numbers from currency/unit suffixes
    num_match = NUMBER_PATTERN.search(candidate)
    if num_match:
        val_str = normalize_numeric_string(num_match.group(0))
        # Look for unit in candidate or on the last line of text
        unit_str = candidate.replace(num_match.group(0), "").replace("\\boxed", "").replace("{", "").replace("}", "").strip()
        last_line = [line.strip() for line in text.splitlines() if line.strip()][-1] if text.splitlines() else ""
        unit_match = re.search(
            r"\b(KZT|PLN|CNY|IDR|PKR|EGP|INR|USD|EUR|тенге|руб|%|percent)\b",
            f"{candidate} {last_line}",
            re.IGNORECASE,
        )
        unit = unit_match.group(0) if unit_match else (unit_str if unit_str else None)
        return val_str, unit

    return candidate, None

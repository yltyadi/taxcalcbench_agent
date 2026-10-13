"""Safe, isolated execution sandbox for Program-Aided CoT (PoT/PAL)."""

import contextlib
import io
import math
import multiprocessing
import traceback
from datetime import date, datetime
from decimal import Decimal
from typing import Any

TIMEOUT_SECONDS = 5.0


def _safe_import(name, *args, **kwargs):
    if name in {"decimal", "math", "datetime", "fractions"}:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Import of module '{name}' is disallowed in sandbox")


def _sandboxed_worker(code_str: str, queue: multiprocessing.Queue):
    """Worker process that executes code in an isolated scope."""
    stdout_capture = io.StringIO()
    allowed_builtins = {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    allowed_globals = {
        "__builtins__": allowed_builtins,
        "Decimal": Decimal,
        "math": math,
        "datetime": datetime,
        "date": date,
    }
    local_vars: dict[str, Any] = {}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, allowed_globals, local_vars)

        ans = local_vars.get("ans") or local_vars.get("answer") or local_vars.get("result")
        if ans is None:
            # Check last printed line
            printed = stdout_capture.getvalue().strip()
            if printed:
                ans = printed.splitlines()[-1].strip()

        # Convert to string to avoid multiprocessing IPC pickling bottlenecks
        ans_str = str(ans) if ans is not None else None
        truncated_stdout = stdout_capture.getvalue()[:4000]

        queue.put({"success": True, "value": ans_str, "stdout": truncated_stdout})
    except Exception as exc:
        queue.put({
            "success": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[:2000],
        })


def execute_python_code(code_str: str, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    """Execute Python code in a separate process with a strict timeout.

    Args:
        code_str: The Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Dict with keys: success (bool), value (Any), error (str or None).
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_sandboxed_worker, args=(code_str, queue))

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        return {
            "success": False,
            "value": None,
            "error": f"TimeoutError: Execution exceeded {timeout} seconds",
        }

    if not queue.empty():
        return queue.get()

    return {
        "success": False,
        "value": None,
        "error": "Process terminated without returning a result",
    }

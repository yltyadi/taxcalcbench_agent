"""Small local-law tools and a bounded decimal expression calculator."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from bisect import bisect_right
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    DecimalException,
    localcontext,
)
from pathlib import Path


def _query_patterns(query: str) -> list[re.Pattern]:
    """Match complete tokens, with optional explicit alphabetic word prefixes."""
    terms = re.findall(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:\*(?!\w))?", query)
    if sum(term.endswith("*") for term in terms) != query.count("*"):
        raise ValueError("Use * only at the end of an alphabetic word stem; numbers must be exact")
    return [re.compile(r"(?<!\w)" + re.escape(term[:-1] if term.endswith("*") else term)
                       + (r"[^\W\d_]*" if term.endswith("*") else "") + r"(?!\w)", re.IGNORECASE)
            for term in terms]


def _relevant_gaps(document: dict, start: int, end: int) -> list[dict]:
    """Return compact notices only for gaps intersecting the supplied excerpt."""
    return [{"label": str(gap.get("label", ""))[:240], "reason": str(gap.get("reason", ""))[:180],
             "start": gap["start"], "end": gap["end"]}
            for gap in document.get("extraction_gaps", []) if isinstance(gap, dict)
            and type(gap.get("start")) is int and type(gap.get("end")) is int
            and gap["start"] < end and gap["end"] > start]


def _plain(value: Decimal) -> str:
    if not value.is_finite() or abs(value.adjusted()) > 1000:
        raise ValueError("Calculation result is outside the supported finite range")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if Decimal(text) == 0 else text


def calculate(expression: str) -> str:
    """Evaluate decimal arithmetic, never Python code.

    Supports +, -, *, /, parentheses, abs, min, max, floor and ceil. Rounding
    functions take (value, decimal_places): round/round_half_up, round_half_even,
    and round_down (toward zero). Division uses 80-digit decimal precision;
    specify the legally required rounding explicitly. Percentages are decimal
    operands such as 0.20, not a percent operator.
    """
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 4000:
        raise ValueError("Provide a nonempty expression of at most 4000 characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid calculator expression") from exc
    if sum(1 for _ in ast.walk(tree)) > 500:
        raise ValueError("Calculator expression is too complex")
    source = expression.strip()

    def visit(node: ast.AST, depth: int = 0) -> Decimal:
        if depth > 40:
            raise ValueError("Calculator expression nesting exceeds 40")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(source, node) or ""
            if not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token):
                raise ValueError("Use plain decimal numbers without separators")
            value = Decimal(token)
            if not value.is_finite() or len(value.as_tuple().digits) > 100 or abs(value.adjusted()) > 100:
                raise ValueError("Calculator operand is outside the supported range")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                result = left / right
            _plain(result)
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if len(node.args) > 100:
                raise ValueError("Too many calculator arguments")
            values = [visit(arg, depth + 1) for arg in node.args]
            if name in {"min", "max"} and values:
                return min(values) if name == "min" else max(values)
            if name == "abs" and len(values) == 1:
                return abs(values[0])
            if name in {"floor", "ceil"} and len(values) == 1:
                return values[0].to_integral_value(rounding=ROUND_FLOOR if name == "floor" else ROUND_CEILING)
            modes = {"round": ROUND_HALF_UP, "round_half_up": ROUND_HALF_UP,
                     "round_half_even": ROUND_HALF_EVEN, "round_down": ROUND_DOWN}
            if name in modes and len(values) == 2:
                places = values[1]
                if places != places.to_integral_value() or not -30 <= places <= 30:
                    raise ValueError("Rounding places must be an integer between -30 and 30")
                return values[0].quantize(Decimal(1).scaleb(-int(places)), rounding=modes[name])
        raise ValueError("Unsupported calculator syntax or function")

    try:
        with localcontext() as context:
            context.prec = 80
            context.Emax, context.Emin = 1000, -1000
            return _plain(visit(tree.body))
    except (DecimalException, OverflowError, RecursionError) as exc:
        raise ValueError("Undefined or out-of-range decimal calculation") from exc


class LawTools:
    """Access only inventory-backed local files; keep per-operation tool receipts."""

    def __init__(self, inventory: dict, source_dir: Path):
        self.source_dir = Path(source_dir).resolve()
        self.documents: dict[str, dict] = {}
        for document in inventory.get("documents", []):
            source_id = document.get("id")
            if not isinstance(source_id, str) or not source_id or source_id in self.documents:
                raise ValueError("Inventory source IDs must be nonempty and unique")
            self.documents[source_id] = document
        self.read_source_ids: set[str] = set()
        self.read_calls = 0
        self.calculation_calls: list[dict[str, str]] = []
        self._text_cache: dict[str, str] = {}
        self._file_stats: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}

    def _path(self, value: object) -> Path:
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("Inventory paths must be relative local paths")
        path = (self.source_dir / value).resolve()
        if not path.is_relative_to(self.source_dir) or path == self.source_dir:
            raise ValueError("Inventory path escapes its source directory")
        return path

    def source_text(self, source_id: str) -> str:
        """Read verified full source text without marking an agent tool read.

        For application normalization/preloading only; expose find_in_laws and
        read_law to models. Missing, modified or unlisted files raise ValueError.
        """
        document = self.documents.get(source_id)
        if document is None:
            raise ValueError("Unknown inventory source ID")
        raw_path, text_path = self._path(document.get("raw_path")), self._path(document.get("text_path"))
        try:
            raw_stat, text_stat = raw_path.stat(), text_path.stat()
            stats = ((raw_stat.st_size, raw_stat.st_mtime_ns), (text_stat.st_size, text_stat.st_mtime_ns))
            if source_id in self._text_cache:
                if stats != self._file_stats[source_id]:
                    raise ValueError("Source files changed during the operation")
                return self._text_cache[source_id]
            raw, text_bytes = raw_path.read_bytes(), text_path.read_bytes()
        except OSError as exc:
            raise ValueError("An inventory source file is unavailable") from exc
        if hashlib.sha256(raw).hexdigest() != document.get("sha256"):
            raise ValueError("Original source checksum does not match the inventory")
        if hashlib.sha256(text_bytes).hexdigest() != document.get("text_sha256"):
            raise ValueError("Extracted source checksum does not match the inventory")
        try:
            text = text_bytes.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Extracted source is not UTF-8 text") from exc
        if not text.strip():
            raise ValueError("Extracted source has no readable text")
        self._text_cache[source_id], self._file_stats[source_id] = text, stats
        return text

    def read_law(self, source_id: str, start_line: int = 1, end_line: int = 160) -> dict:
        """Read exact local text at 1-based inclusive line numbers (at most 400 lines).

        Start with inventory source IDs. Returned text preserves saved source
        whitespace; use its actual wording for Supporting passage citations.
        """
        if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
            raise ValueError("Line bounds must be positive ordered integers")
        if end_line - start_line + 1 > 400:
            raise ValueError("Read at most 400 lines per call")
        lines = self.source_text(source_id).splitlines(keepends=True)
        if start_line > len(lines):
            raise ValueError("Start line is beyond the source; use the returned line count")
        end_line = min(end_line, len(lines))
        excerpt = "".join(lines[start_line - 1:end_line])
        if len(excerpt) > 40000:
            raise ValueError("Requested excerpt exceeds 40000 characters; request fewer lines")
        self.read_source_ids.add(source_id)
        self.read_calls += 1
        document = self.documents[source_id]
        start_char = sum(len(line) for line in lines[:start_line - 1])
        return {"source_id": source_id, "title": str(document.get("title", ""))[:256],
                "url": document.get("final_url") or document.get("url"),
                "start_line": start_line, "end_line": end_line, "total_lines": len(lines), "text": excerpt,
                "partial": document.get("partial", False),
                "extraction_gaps": _relevant_gaps(document, start_char, start_char + len(excerpt))}

    def list_sections(self, query: str, source_id: str) -> dict:
        """List up to 40 recorded provision headings and their inclusive line ranges.

        Search heading words in any order; all must match, case insensitively.
        Add * to an alphabetic stem to match inflections, e.g. подоходн* налог*.
        Otherwise words and numbers match exactly. If an article number is
        known, search that number alone rather than adding a guessed title.
        An empty query lists the first headings. These are
        navigation metadata: use read_law to inspect the underlying legal text.
        Unavailable provisions are marked and cannot support legal citations.
        """
        if not isinstance(query, str) or len(query) > 300:
            raise ValueError("Section query must be a string of at most 300 characters")
        patterns = _query_patterns(query)
        if query.strip() and not patterns:
            raise ValueError("Section query needs words or numbers, or an empty string")
        text = self.source_text(source_id)
        document = self.documents[source_id]
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        matches = []
        for section in document.get("boundaries", []):
            if not isinstance(section, dict) or section.get("kind") != "section":
                continue
            label, start, end = section.get("label"), section.get("start"), section.get("end")
            if not isinstance(label, str) or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("Inventory section boundary is invalid")
            if all(pattern.search(label) for pattern in patterns):
                matches.append({"label": label, "start_line": bisect_right(offsets, start),
                                "end_line": bisect_right(offsets, end - 1),
                                "available": not bool(_relevant_gaps(document, start, end))})
        result = {"source_id": source_id, "sections": [], "total_matches": len(matches), "truncated": False}
        for section in matches[:40]:
            candidate = {**result, "sections": [*result["sections"], section], "truncated": True}
            if len(json.dumps(candidate, ensure_ascii=False)) > 10000:
                break
            result["sections"].append(section)
        result["truncated"] = len(result["sections"]) < len(matches)
        return result

    def read_sections(self, source_id: str, queries: list[str]) -> list[dict]:
        """Read several uniquely matched provisions from one saved official file.

        Use article numbers or heading queries accepted by list_sections. Each
        result retains section locators and includes exact read_law text and
        provenance only when the whole matched range is readable and fits its
        existing limits. An issue explains ambiguous, missing or large ranges;
        narrow those with list_sections/read_law. Source integrity errors raise.
        """
        if not isinstance(queries, list) or not queries:
            raise ValueError("Provide a nonempty list of section queries")
        results = []
        for query in queries:
            result = {"query": query, **self.list_sections(query, source_id), "issue": None}
            sections = result["sections"]
            if result["total_matches"] == 0:
                result["issue"] = "No recorded section matches; use find_in_laws or refine the heading query"
            elif result["total_matches"] != 1 or result["truncated"]:
                result["issue"] = "Section query is ambiguous; refine it or inspect the returned locators"
            elif not sections[0]["available"]:
                result["issue"] = "Section contains unavailable source text and cannot support citations"
            else:
                section = sections[0]
                try:
                    excerpt = self.read_law(source_id, section["start_line"], section["end_line"])
                except ValueError as error:
                    if str(error) not in {"Read at most 400 lines per call",
                                          "Requested excerpt exceeds 40000 characters; request fewer lines"}:
                        raise
                    result["issue"] = f"{error}; use read_law with narrower line bounds from the returned locator"
                else:
                    if excerpt["extraction_gaps"]:
                        result["issue"] = "Section line range overlaps unavailable text and cannot be read as a complete provision"
                    else:
                        result.update(excerpt)
            results.append(result)
        return results

    def find_in_laws(self, query: str, source_ids: list[str] | None = None) -> list[dict]:
        """Find up to six compact exact-text excerpts, best matching phrases first.

        Use short phrases; add * to alphabetic word stems for inflections, e.g.
        подоходн* налог*. Ordinary words and numbers match exactly. Search a
        known article number alone, without a guessed title. Same-line matches
        precede matches across three lines. Source IDs optionally narrow the
        search. Open returned line numbers with read_law for more context.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            raise ValueError("Search needs a nonempty query of at most 300 characters")
        patterns = _query_patterns(query)
        if not patterns:
            raise ValueError("Search needs words or numbers")
        phrase = None if "*" in query else re.compile(
            r"(?<!\w)" + r"\s+".join(re.escape(part) for part in query.split()) + r"(?!\w)", re.IGNORECASE)
        if source_ids is None:
            selected = list(self.documents)
        elif not isinstance(source_ids, list) or not source_ids or any(
            not isinstance(key, str) or key not in self.documents for key in source_ids
        ):
            raise ValueError("Search source IDs must be a nonempty list of inventory IDs")
        else:
            selected = list(dict.fromkeys(source_ids))
        candidates = []
        texts, offsets_by_source = {}, {}
        for source_order, source_id in enumerate(selected):
            text = self.source_text(source_id)
            lines = text.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line))
            texts[source_id], offsets_by_source[source_id] = text, offsets
            for index in range(len(lines)):
                line = lines[index]
                exact = phrase.search(line) if phrase else None
                if exact:
                    rank, core_end, match_offset = 0, index + 1, exact.start()
                elif all(pattern.search(line) for pattern in patterns):
                    rank, core_end, match_offset = 1, index + 1, min(pattern.search(line).start() for pattern in patterns)
                else:
                    core_end = min(len(lines), index + 3)
                    window = "".join(lines[index:core_end])
                    if not all(pattern.search(window) for pattern in patterns):
                        continue
                    rank, match_offset = 2, min(pattern.search(window).start() for pattern in patterns)
                start, end = offsets[index], offsets[core_end]
                if end - start > 1200:
                    start = max(start, start + match_offset - 100)
                    end = min(offsets[core_end], start + 1200)
                else:
                    # Add a little adjacent context only when the complete
                    # lines fit, keeping each returned quote contiguous.
                    if index > 0 and end - offsets[index - 1] <= 1200:
                        start = offsets[index - 1]
                    if core_end < len(lines) and offsets[core_end + 1] - start <= 1200:
                        end = offsets[core_end + 1]
                if not all(pattern.search(text[start:end]) for pattern in patterns):
                    continue
                candidates.append((rank, source_order, index, source_id, start, end))
        found, selected_spans = [], {}
        for _, _, _, source_id, start, end in sorted(candidates):
            if any(start < old_end and end > old_start for old_start, old_end in selected_spans.get(source_id, [])):
                continue
            text, offsets = texts[source_id], offsets_by_source[source_id]
            document = self.documents[source_id]
            hit = {"source_id": source_id, "title": str(document.get("title", ""))[:256],
                   "start_line": bisect_right(offsets, start), "end_line": bisect_right(offsets, end - 1),
                   "start_char": start, "end_char": end, "text": text[start:end],
                   "partial": document.get("partial", False), "extraction_gaps": _relevant_gaps(document, start, end)}
            if len(json.dumps([*found, hit], ensure_ascii=False)) > 10000:
                continue
            found.append(hit)
            selected_spans.setdefault(source_id, []).append((start, end))
            self.read_source_ids.add(source_id)
            if len(found) == 6:
                break
        return found

    def calculate(self, expression: str) -> str:
        """Evaluate +, -, *, / and parentheses with 80-digit decimal arithmetic.

        Supports min, max, abs, floor and ceil. round(value, places) and
        round_half_up(value, places) use half-up rounding; round_half_even and
        round_down are also available. Choose the legally required mode. Write
        percentages as decimal operands (0.20). No arbitrary code is executed.
        """
        result = calculate(expression)
        self.calculation_calls.append({"expression": expression, "result": result})
        return result

    def calculate_many(self, expressions: list[str]) -> list[dict[str, str]]:
        """Evaluate 1 to 10 decimal expressions in one call, preserving their order.

        Uses the same operations and explicit rounding modes as calculate.
        Each successful calculation receives its usual individual tool receipt.
        An invalid expression raises an error; no partial result list is returned.
        """
        if not isinstance(expressions, list) or not 1 <= len(expressions) <= 10:
            raise ValueError("Provide a list of 1 to 10 calculator expressions")
        return [{"expression": expression, "result": self.calculate(expression)} for expression in expressions]

"""Offline synthetic-tool tests; no real laws or network calls."""

import json
from hashlib import sha256

import pytest

from taxcalcbench.tools import LawTools, calculate


@pytest.mark.parametrize("expression,answer", [
    ("0.1 + 0.2", "0.3"), ("(1000 - 200) * 0.15", "120"), ("-5 + +2", "-3"),
    ("round(2.675, 2)", "2.68"), ("round_half_even(2.665, 2)", "2.66"),
    ("round_half_up(-2.665, 2)", "-2.67"), ("round_down(-2.669, 2)", "-2.66"),
    ("round_half_up(1234, -2)", "1200"), ("min(4, 7, 2) + max(1, 3)", "5"),
    ("floor(-1.2) + ceil(1.2)", "0"), ("abs(-2.5)", "2.5"), ("round(1 / 3, 6)", "0.333333"),
    ("9007199254740993.01 - 9007199254740993", "0.01"), ("1e3 / 4", "250"),
])
def test_exact_decimal_arithmetic_and_explicit_rounding(expression, answer):
    assert calculate(expression) == answer


@pytest.mark.parametrize("expression", [
    "__import__('os').getcwd()", "open('secret')", "(1).__class__", "[1][0]", "sum([1, 2])",
    "True + 1", "'10'", "lambda: 1", "2 ** 100000", "1 / 0", "1 % 2", "1_000 + 1", "0xff",
    "round(1.25, digits=2)", "round(1.25, 2.5)", "round(1.25, 100)", "1e999999999", "", "[x for x in []]",
])
def test_unsafe_undefined_or_unbounded_calculations_reject(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculator_limits_expression_size_and_complexity():
    with pytest.raises(ValueError):
        calculate("1+" * 2200 + "1")
    with pytest.raises(ValueError):
        calculate("1+" * 260 + "1")


@pytest.fixture
def laws(tmp_path):
    text = "SYNTHETIC NOT-LAW\r\nTest article 1\r\nThe toy\u00a0rate is 0.10.\r\nEnd.\r\n"
    (tmp_path / "original.bin").write_bytes(b"SYNTHETIC original, not law")
    (tmp_path / "law.txt").write_bytes(text.encode())
    inventory = {"documents": [{"id": "toy", "title": "SYNTHETIC test law", "url": "https://synthetic.invalid/toy",
        "final_url": "https://synthetic.invalid/toy", "raw_path": "original.bin", "text_path": "law.txt",
        "sha256": sha256(b"SYNTHETIC original, not law").hexdigest(), "text_sha256": sha256(text.encode()).hexdigest(),
        "country": "ZZ", "language": "ru", "kind": "law", "partial": True,
        "extraction_gaps": [{"label": "SYNTHETIC related notice", "reason": "Synthetic test only", "start": 0, "end": len(text)}]}]}
    return LawTools(inventory, tmp_path), inventory, tmp_path, text


def test_read_uses_exact_source_lines_and_tracks_only_agent_reads(laws):
    tools, _, _, text = laws
    assert tools.source_text("toy") == text
    assert tools.read_source_ids == set()
    assert tools.read_calls == 0
    result = tools.read_law("toy", 2, 3)
    assert result["text"] == "Test article 1\r\nThe toy\u00a0rate is 0.10.\r\n"
    assert result["start_line"] == 2 and result["end_line"] == 3 and result["total_lines"] == 4
    assert tools.read_source_ids == {"toy"}
    assert tools.read_calls == 1
    assert result["partial"] and result["extraction_gaps"]
    assert tools.read_law("toy")["text"] == text


@pytest.mark.parametrize("start,end", [(0, 1), (3, 2), (1, 401), (5, 6), (True, 3), (1, 2.5)])
def test_read_rejects_invalid_line_ranges(laws, start, end):
    with pytest.raises(ValueError):
        laws[0].read_law("toy", start, end)
    assert laws[0].read_calls == 0


def test_local_search_returns_source_text_and_tracks_provided_excerpts(laws):
    tools, _, _, text = laws
    assert tools.find_in_laws("nonexistent keyword") == []
    assert tools.read_source_ids == set()
    result = tools.find_in_laws("toy rate", ["toy"])
    assert result and result[0]["text"] in text
    assert result[0]["source_id"] == "toy" and result[0]["partial"]
    assert tools.read_source_ids == {"toy"}
    for query, ids in [("", None), ("...", None), ("toy", []), ("toy", ["unlisted"])]:
        with pytest.raises(ValueError):
            tools.find_in_laws(query, ids)


def test_calculator_receipts_record_only_successful_agent_tool_calls(laws):
    tools = laws[0]
    assert calculate("2 * 3") == "6" and tools.calculation_calls == []
    assert tools.calculate("2 * 3") == "6"
    assert tools.calculation_calls == [{"expression": "2 * 3", "result": "6"}]
    with pytest.raises(ValueError):
        tools.calculate("1 / 0")
    assert len(tools.calculation_calls) == 1


def test_calculate_many_preserves_results_order_and_individual_receipts(laws):
    tools = laws[0]
    result = tools.calculate_many(["0.1 + 0.2", "round(2.675, 2)", "100 * 0.15"])
    assert result == [{"expression": "0.1 + 0.2", "result": "0.3"},
                      {"expression": "round(2.675, 2)", "result": "2.68"},
                      {"expression": "100 * 0.15", "result": "15"}]
    assert tools.calculation_calls == result


@pytest.mark.parametrize("expressions", [[], ["1"] * 11, "1 + 2", None])
def test_calculate_many_rejects_invalid_batch_bounds(laws, expressions):
    with pytest.raises(ValueError, match="1 to 10"):
        laws[0].calculate_many(expressions)
    assert laws[0].calculation_calls == []


def test_calculate_many_accepts_upper_bound_and_rejects_invalid_expressions(laws):
    tools = laws[0]
    assert len(tools.calculate_many(["1 + 1"] * 10)) == 10
    prior = tools.calculation_calls.copy()
    with pytest.raises(ValueError):
        tools.calculate_many(["1 / 0", "unsupported()"])
    assert tools.calculation_calls == prior
    with pytest.raises(ValueError):
        tools.calculate_many(["2 + 3", "unsupported()"])
    assert tools.calculation_calls == [*prior, {"expression": "2 + 3", "result": "5"}]


@pytest.mark.parametrize("field", ["sha256", "text_sha256"])
def test_source_hashes_are_verified(laws, field):
    tools, inventory, _, _ = laws
    inventory["documents"][0][field] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        tools.source_text("toy")


def test_changed_cached_source_rejects(laws):
    tools, _, folder, _ = laws
    tools.source_text("toy")
    (folder / "law.txt").write_text("CHANGED SYNTHETIC CONTENT", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        tools.source_text("toy")


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd"])
def test_inventory_cannot_read_outside_source_directory(laws, path):
    tools, inventory, _, _ = laws
    inventory["documents"][0]["text_path"] = path
    with pytest.raises(ValueError, match="relative|escapes"):
        tools.source_text("toy")


def test_symlink_outside_source_directory_rejects(laws, tmp_path):
    tools, inventory, folder, _ = laws
    outside = folder.parent / "synthetic-outside.txt"
    outside.write_text("SYNTHETIC outside source tree", encoding="utf-8")
    (folder / "escape.txt").symlink_to(outside)
    inventory["documents"][0]["text_path"] = "escape.txt"
    with pytest.raises(ValueError, match="escapes"):
        tools.source_text("toy")


def test_unknown_or_duplicate_inventory_ids_reject(laws):
    tools, inventory, folder, _ = laws
    with pytest.raises(ValueError, match="Unknown"):
        tools.source_text("unknown")
    inventory["documents"].append(inventory["documents"][0].copy())
    with pytest.raises(ValueError, match="unique"):
        LawTools(inventory, folder)


def replace_law(laws, text, gaps=None):
    _, inventory, folder, _ = laws
    (folder / "law.txt").write_bytes(text.encode())
    inventory["documents"][0]["text_sha256"] = sha256(text.encode()).hexdigest()
    inventory["documents"][0]["extraction_gaps"] = gaps or []
    inventory["documents"][0]["partial"] = bool(gaps)
    return LawTools(inventory, folder)


def test_search_ranks_exact_phrase_before_same_line_then_loose_windows(laws):
    text = "toy\nrate\n\n\n\n\ntoy conditional rate\n\n\n\n\ntoy rate exactly\n"
    tools = replace_law(laws, text)
    results = tools.find_in_laws("toy rate")
    assert "toy rate exactly" in results[0]["text"]
    assert "toy conditional rate" in results[1]["text"]
    assert "toy\nrate" in results[2]["text"]


def test_search_does_not_match_article_number_substrings(laws):
    text = "Article 357: wrong target\n\n\n\nArticle 35: correct target\n\n\n\nArticle 135: wrong target\n"
    tools = replace_law(laws, text)
    results = tools.find_in_laws("35")
    assert len(results) == 1 and "Article 35:" in results[0]["text"]
    assert tools.find_in_laws("35", ["toy"]) == results


def test_search_limits_total_serialized_size_and_removes_overlap(laws):
    text = "".join(f"Article {index}: target provision " + "x" * 900 + "\n\n\n" for index in range(30))
    tools = replace_law(laws, text)
    tools.documents["toy"]["title"] = "T" * 20000
    results = tools.find_in_laws("target provision")
    assert len(results) == 6
    assert len(json.dumps(results, ensure_ascii=False)) <= 10000
    for hit in results:
        assert hit["text"] == text[hit["start_char"]:hit["end_char"]]
        assert len(hit["text"]) <= 1200
    spans = sorted((hit["start_char"], hit["end_char"]) for hit in results)
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_search_and_read_return_only_intersecting_gap_metadata(laws):
    text = "target available clause\n\n\n\n[UNAVAILABLE PROVISION: OTHER]\n"
    boundary = text.index("[UNAVAILABLE")
    gap = {"label": "OTHER", "reason": "Synthetic missing formula", "start": boundary, "end": len(text)}
    tools = replace_law(laws, text, [gap])
    results = tools.find_in_laws("target")
    assert results[0]["partial"] and results[0]["extraction_gaps"] == []
    assert tools.read_law("toy", 1, 1)["extraction_gaps"] == []
    assert tools.read_law("toy", 5, 5)["extraction_gaps"] == [gap]
    marker_results = tools.find_in_laws("UNAVAILABLE PROVISION")
    assert marker_results[0]["extraction_gaps"] == [gap]


def section_law(laws, headings):
    text = "SYNTHETIC NOT-LAW\r\n"
    boundaries = []
    for heading in headings:
        start = len(text)
        text += heading + "\r\nSynthetic provision body.\r\n"
        boundaries.append({"kind": "section", "label": heading, "start": start, "end": len(text)})
    tools = replace_law(laws, text)
    tools.documents["toy"]["boundaries"] = boundaries
    return tools


def test_section_listing_matches_complete_heading_words_and_numbers(laws):
    tools = section_law(laws, ["Статья 357. Тестовый налог", "Статья 35. Тестовый налог", "Статья 135. Иной налог"])
    result = tools.list_sections("НАЛОГ 35", "toy")
    assert result == {"source_id": "toy", "sections": [
        {"label": "Статья 35. Тестовый налог", "start_line": 4, "end_line": 5, "available": True}],
        "total_matches": 1, "truncated": False}
    assert tools.list_sections("тестов", "toy")["sections"] == []
    assert tools.list_sections("999", "toy")["total_matches"] == 0
    assert tools.read_source_ids == set()
    section = result["sections"][0]
    assert tools.read_law("toy", section["start_line"], section["end_line"])["text"].startswith(section["label"])


def test_empty_section_query_returns_bounded_first_headings(laws):
    tools = section_law(laws, [f"Article {number}. Synthetic heading" for number in range(55)])
    result = tools.list_sections("", "toy")
    assert len(result["sections"]) == 40
    assert result["sections"][0]["label"] == "Article 0. Synthetic heading"
    assert result["sections"][-1]["label"] == "Article 39. Synthetic heading"
    assert result["total_matches"] == 55 and result["truncated"]
    assert len(json.dumps(result, ensure_ascii=False)) <= 10000


def test_section_listing_marks_gaps_and_ignores_nonsection_boundaries(laws):
    tools = section_law(laws, ["Article 1. Available", "Article 2. Missing formula"])
    document = tools.documents["toy"]
    missing = document["boundaries"][1]
    document["extraction_gaps"] = [{**missing, "reason": "Synthetic unsupported formula"}]
    document["boundaries"].append({"kind": "page", "label": "Page 1", "start": 0, "end": 10})
    result = tools.list_sections("", "toy")
    assert [section["available"] for section in result["sections"]] == [True, False]
    assert result["total_matches"] == 2


def test_section_listing_checks_inventory_and_query(laws):
    tools = laws[0]
    assert tools.list_sections("", "toy")["sections"] == []
    for query in [None, "x" * 301, "..."]:
        with pytest.raises(ValueError):
            tools.list_sections(query, "toy")
    with pytest.raises(ValueError, match="Unknown"):
        tools.list_sections("", "unknown")
    tools.documents["toy"]["boundaries"] = [{"kind": "section", "label": "Invalid", "start": 0, "end": 9999}]
    with pytest.raises(ValueError, match="boundary"):
        tools.list_sections("", "toy")


def test_section_listing_caps_all_serialized_metadata(laws):
    tools = section_law(laws, [f"Article {number}. " + "Synthetic " * 100 for number in range(40)])
    result = tools.list_sections("", "toy")
    assert 0 < len(result["sections"]) < 40 and result["truncated"]
    assert len(json.dumps(result, ensure_ascii=False)) <= 10000


def test_explicit_word_prefixes_find_inflections_but_keep_numbers_exact(laws):
    tools = section_law(laws, ["Статья 35. Исчисление подоходного налога",
                                "Статья 357. Подоходный налог", "Статья 135. Неподоходный налог"])
    result = tools.list_sections("ПОДОХОДН* НАЛОГ*", "toy")
    assert result["total_matches"] == 2
    assert tools.list_sections("подоходный налог", "toy")["total_matches"] == 1
    result = tools.list_sections("35 подоходн* налог*", "toy")
    assert result["total_matches"] == 1 and result["sections"][0]["label"].startswith("Статья 35.")
    hits = tools.find_in_laws("35 подоходн* налог*", ["toy"])
    assert len(hits) == 1 and "Исчисление подоходного налога" in hits[0]["text"]
    assert hits[0]["text"] == tools.source_text("toy")[hits[0]["start_char"]:hits[0]["end_char"]]


@pytest.mark.parametrize("query", ["35*", "*налог", "нал**", "нал*ог", "*", "налог*35"])
def test_only_alphabetic_trailing_wildcards_are_supported(laws, query):
    tools = laws[0]
    for search in [lambda: tools.list_sections(query, "toy"), lambda: tools.find_in_laws(query)]:
        with pytest.raises(ValueError, match="alphabetic"):
            search()


def test_prefix_searches_preserve_result_count_and_size_bounds(laws):
    tools = section_law(laws, [f"Article {number}. Подоходного налога" for number in range(60)])
    sections = tools.list_sections("подоходн* налог*", "toy")
    assert sections["total_matches"] == 60 and len(sections["sections"]) == 40
    assert sections["truncated"] and len(json.dumps(sections, ensure_ascii=False)) <= 10000
    hits = tools.find_in_laws("подоходн* налог*")
    assert 0 < len(hits) <= 6 and len(json.dumps(hits, ensure_ascii=False)) <= 10000


def test_read_sections_batches_exact_provisions_in_requested_order_with_provenance(laws):
    tools = section_law(laws, ["Статья 35. Исчисление подоходного налога", "Статья 357. Иной налог"])
    result = tools.read_sections("toy", ["357", "35 подоходн* налог*"])
    assert [item["query"] for item in result] == ["357", "35 подоходн* налог*"]
    assert [item["start_line"] for item in result] == [4, 2]
    assert [item["end_line"] for item in result] == [5, 3]
    text = tools.source_text("toy")
    for item, boundary in zip(result, reversed(tools.documents["toy"]["boundaries"])):
        assert item["issue"] is None and item["total_matches"] == 1 and not item["truncated"]
        assert item["text"] == text[boundary["start"]:boundary["end"]]
        assert item["text"].endswith("Synthetic provision body.\r\n")
        assert item["source_id"] == "toy" and item["title"] == "SYNTHETIC test law"
        assert item["url"] == "https://synthetic.invalid/toy" and item["total_lines"] == 5
        assert item["extraction_gaps"] == []
    assert tools.read_source_ids == {"toy"} and tools.read_calls == 2


def test_read_sections_does_not_choose_first_ambiguous_or_truncated_match(laws):
    tools = section_law(laws, [f"Article {number}. Synthetic heading" for number in range(55)])
    results = tools.read_sections("toy", ["heading", "999", "35"])
    ambiguous, absent, exact = results
    assert ambiguous["total_matches"] == 55 and ambiguous["truncated"]
    assert len(ambiguous["sections"]) == 40 and "ambiguous" in ambiguous["issue"]
    assert "text" not in ambiguous
    assert absent["sections"] == [] and absent["total_matches"] == 0
    assert "No recorded section" in absent["issue"] and "text" not in absent
    assert exact["text"].startswith("Article 35.") and exact["issue"] is None
    assert tools.read_calls == 1


def test_read_sections_reports_unavailable_provision_without_returning_its_text(laws):
    tools = section_law(laws, ["Article 1. Available", "Article 2. Missing formula"])
    missing = tools.documents["toy"]["boundaries"][1]
    tools.documents["toy"]["extraction_gaps"] = [{**missing, "reason": "Synthetic unsupported formula"}]
    results = tools.read_sections("toy", ["2", "1"])
    assert results[0]["sections"][0]["available"] is False
    assert "unavailable" in results[0]["issue"] and "text" not in results[0]
    assert results[1]["issue"] is None and results[1]["text"].startswith("Article 1.")
    assert tools.read_calls == 1


@pytest.mark.parametrize("body", ["Synthetic line\n" * 400, "x" * 40000 + "\n"])
def test_read_sections_returns_oversized_locator_without_silently_truncating(laws, body):
    text = "Article 1. Synthetic large provision\n" + body + "Article 2. Small\nReadable.\n"
    split = text.index("Article 2.")
    tools = replace_law(laws, text)
    tools.documents["toy"]["boundaries"] = [
        {"kind": "section", "label": "Article 1. Synthetic large provision", "start": 0, "end": split},
        {"kind": "section", "label": "Article 2. Small", "start": split, "end": len(text)},
    ]
    oversized, readable = tools.read_sections("toy", ["1", "2"])
    assert oversized["sections"][0]["start_line"] == 1
    assert oversized["sections"][0]["end_line"] == len(text[:split].splitlines())
    assert "read_law" in oversized["issue"] and "text" not in oversized
    assert readable["text"] == text[split:] and readable["issue"] is None
    assert tools.read_calls == 1
    assert tools.read_law("toy", 1, 1)["text"] == "Article 1. Synthetic large provision\n"


def test_read_sections_does_not_return_unavailable_adjacent_text_on_shared_line(laws):
    text = "Article 1. Readable text. [UNAVAILABLE ADJACENT PROVISION]\n"
    end = text.index("[UNAVAILABLE")
    gap = {"label": "Adjacent synthetic provision", "reason": "Synthetic gap", "start": end, "end": len(text)}
    tools = replace_law(laws, text, [gap])
    tools.documents["toy"]["boundaries"] = [
        {"kind": "section", "label": "Article 1. Readable text", "start": 0, "end": end}]
    result = tools.read_sections("toy", ["1"])[0]
    assert "overlaps unavailable" in result["issue"] and "text" not in result


@pytest.mark.parametrize("field", ["sha256", "text_sha256"])
def test_read_sections_propagates_source_checksum_errors(laws, field):
    tools = section_law(laws, ["Article 1. Synthetic section"])
    tools.documents["toy"][field] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        tools.read_sections("toy", ["1"])
    assert tools.read_calls == 0


def test_read_sections_propagates_tamper_between_lookup_and_read(laws, monkeypatch):
    tools = section_law(laws, ["Article 1. Synthetic section"])
    original = tools.read_law

    def changed(*args):
        (laws[2] / "law.txt").write_text("CHANGED SYNTHETIC CONTENT")
        return original(*args)

    monkeypatch.setattr(tools, "read_law", changed)
    with pytest.raises(ValueError, match="changed"):
        tools.read_sections("toy", ["1"])
    assert tools.read_calls == 0


@pytest.mark.parametrize("queries", [[], "1", None, [None], ["..."], ["x" * 301]])
def test_read_sections_rejects_invalid_queries(laws, queries):
    with pytest.raises(ValueError):
        laws[0].read_sections("toy", queries)


def test_read_sections_rejects_unknown_source(laws):
    with pytest.raises(ValueError, match="Unknown"):
        laws[0].read_sections("unknown", ["1"])

"""Statutory section and article-aware document chunker for RAG."""

import json
import re
from pathlib import Path
from typing import Any

UNIVERSAL_HEADING_PATTERN = re.compile(
    r"^(?:[ \t\u00a0]*(?:Статья|СТАТЬЯ|Бап|БАП|Article|ARTICLE|Section|SECTION|Pasal|PASAL|Art\.|المادة|مادة|第[一二三四五六七八九十百千万0-9]+条)[ \t\u00a0.:()0-9٠-٩\-]+)",
    re.MULTILINE,
)


def chunk_document_text(
    doc_id: str,
    title: str,
    text: str,
    heading_regex: re.Pattern | None = None,
    max_chars: int = 2400,
    overlap_chars: int = 300,
) -> list[dict[str, Any]]:
    """Split a statutory document into article/section-aligned chunks."""
    pattern = heading_regex or UNIVERSAL_HEADING_PATTERN
    matches = list(pattern.finditer(text))

    sections: list[str] = []
    if not matches:
        sections = [text]
    else:
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(preamble)
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()
            if sec_text:
                sections.append(sec_text)

    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    for sec in sections:
        if len(sec) <= max_chars:
            chunks.append({
                "chunk_id": f"{doc_id}_c{chunk_idx}",
                "source_id": doc_id,
                "title": title,
                "text": sec,
            })
            chunk_idx += 1
        else:
            # Sliding window over oversized articles while preserving header line
            first_line = sec.splitlines()[0][:180] if sec.splitlines() else ""
            step = max(max_chars - overlap_chars, 500)
            for offset in range(0, len(sec), step):
                window = sec[offset : offset + max_chars].strip()
                if not window:
                    continue
                if offset > 0 and first_line and not window.startswith(first_line):
                    window = f"{first_line} (cont.)\n{window}"
                chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "source_id": doc_id,
                    "title": title,
                    "text": window,
                })
                chunk_idx += 1
                if offset + max_chars >= len(sec):
                    break

    return chunks


def load_and_chunk_corpus(
    source_dir: Path,
    max_chars: int = 2400,
    overlap_chars: int = 300,
) -> list[dict[str, Any]]:
    """Load all official documents from source_dir and chunk them."""
    sources_json = source_dir / "sources.json"
    if not sources_json.exists():
        raise FileNotFoundError(f"Missing sources inventory: {sources_json}")

    with open(sources_json, encoding="utf-8") as f:
        inventory = json.load(f)

    policy = inventory.get("source_policy", {})
    raw_pattern = policy.get("section_heading_pattern")
    heading_regex = None
    if raw_pattern:
        try:
            heading_regex = re.compile(raw_pattern, re.MULTILINE)
        except re.error:
            heading_regex = UNIVERSAL_HEADING_PATTERN

    all_chunks: list[dict[str, Any]] = []
    for doc in inventory.get("documents", []):
        if doc.get("kind") == "navigation":
            continue
        doc_id = doc["id"]
        title = doc.get("title", doc_id)
        text_rel = doc.get("text_path") or f"text/{doc_id}.txt"
        text_path = source_dir / text_rel
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        doc_chunks = chunk_document_text(
            doc_id=doc_id,
            title=title,
            text=text,
            heading_regex=heading_regex,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks

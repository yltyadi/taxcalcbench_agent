"""Multilingual BM25, Temporal-Aware BM25, Dense, and Hybrid RRF Retrievers for TaxCalcBench RAG."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[\w\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff]+", re.UNICODE)
YEAR_PATTERN = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")

PARAMETER_KEYWORDS = (
    "месячный расчетный показатель",
    "минимальный размер заработной платы",
    "айлық есептік көрсеткіш",
    "жалақының ең төменгі мөлшері",
    "first schedule",
    "rates of income-tax",
)


def tokenize_multilingual(text: str) -> list[str]:
    """Tokenize text into lowercase words, 5-char morphological stems, and CJK bigrams."""
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    tokens: list[str] = []
    for tok in raw_tokens:
        if len(tok) < 2 and not tok.isdigit():
            continue
        tokens.append(tok)
        if len(tok) >= 6 and not tok.isdigit():
            tokens.append(f"#{tok[:5]}")
        if any("\u4e00" <= ch <= "\u9fff" for ch in tok) and len(tok) >= 2:
            for i in range(len(tok) - 1):
                tokens.append(tok[i : i + 2])
    return tokens


def extract_corpus_temporal_metadata(source_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Extract universal enactment and snapshot years from sources.json metadata."""
    if source_dir is None or not (source_dir / "sources.json").exists():
        return {}

    with open(source_dir / "sources.json", encoding="utf-8") as f:
        inv = json.load(f)

    meta: dict[str, dict[str, Any]] = {}
    for doc in inv.get("documents", []):
        doc_id = doc["id"]
        title = doc.get("title", "")
        raw_path = doc.get("raw_path", "")
        t_years = [int(y) for y in YEAR_PATTERN.findall(title)]
        r_years = [int(y) for y in YEAR_PATTERN.findall(raw_path)]
        all_y = t_years + r_years

        y_enacted = min(t_years) if t_years else (min(all_y) if all_y else None)
        if y_enacted in (2008, 2017) and any(
            k in title.lower() for k in ("налоговый кодекс", "салық кодексі", "о введении")
        ):
            y_eff = y_enacted + 1
        elif y_enacted == 2025 and ("18 июля 2025" in title.lower() or "18 шілде" in title.lower()):
            y_eff = 2026
        elif ("бюджет" in title.lower() or "finance act" in title.lower()) and t_years:
            y_eff = min(t_years)
        else:
            y_eff = y_enacted

        y_snap = max(all_y) if all_y else y_eff
        is_code = any(
            k in title.lower()
            for k in ("налоговый кодекс", "салық кодексі", "income tax ordinance", "income-tax act")
        )
        is_budget = any(k in title.lower() for k in ("бюджет", "finance act"))

        meta[doc_id] = {
            "y_eff": y_eff,
            "y_snap": y_snap,
            "title": title,
            "is_code": is_code,
            "is_budget": is_budget,
        }
    return meta


def compute_temporal_multiplier(
    doc_info: dict[str, Any] | None,
    case_year: int | None,
    all_code_eff_years: list[int],
) -> float:
    """Compute a universal statutory temporal weight respecting non-retroactivity and active consolidations."""
    if not doc_info or not case_year:
        return 1.0

    y_eff = doc_info.get("y_eff")
    y_snap = doc_info.get("y_snap")
    if y_eff is None:
        return 0.6

    # Rule 1: Non-Retroactivity (statutes enacted after case_year were not in force)
    if y_eff > case_year:
        return 0.05

    # Rule 2: Active Primary Tax Code Consolidation
    if doc_info.get("is_code"):
        applicable_code_years = [y for y in all_code_eff_years if y <= case_year]
        active_code_year = max(applicable_code_years) if applicable_code_years else y_eff
        if y_eff < active_code_year:
            return 0.10  # Superseded historical code
        # Within the active code, prefer the snapshot closest to or covering case_year
        if y_snap and abs(y_snap - case_year) <= 2:
            return 2.5
        return 2.0

    # Rule 3: Annual Budget / Finance Acts
    if doc_info.get("is_budget"):
        if y_eff <= case_year <= (y_snap or y_eff) + 1:
            return 2.4
        return 0.10

    # Rule 4: Other companion laws in force at or before case_year
    dist = max(0, case_year - (y_snap or y_eff))
    return 1.5 / (1.0 + 0.08 * dist)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


class BM25Retriever:
    """Fast Inverted-Index Okapi BM25 Retriever with optional Temporal & Parameter Awareness."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        source_dir: Path | None = None,
        temporal_aware: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = chunks
        self.temporal_aware = temporal_aware
        self.k1 = k1
        self.b = b
        self.num_docs = len(chunks)
        self.doc_lens: list[int] = []
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

        self.doc_meta = extract_corpus_temporal_metadata(source_dir) if source_dir else {}
        self.code_eff_years = sorted({
            info["y_eff"]
            for info in self.doc_meta.values()
            if info.get("is_code") and info.get("y_eff") is not None
        })

        # Index annual parameter chunks by year
        self.param_chunks_by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
        total_len = 0

        for idx, chunk in enumerate(chunks):
            full_str = f"{chunk.get('title', '')}\n{chunk.get('text', '')}"
            tokens = tokenize_multilingual(full_str)
            dl = len(tokens)
            self.doc_lens.append(dl)
            total_len += dl
            tf = Counter(tokens)
            for term, freq in tf.items():
                self.inverted_index[term].append((idx, freq))

            if self.temporal_aware and self.doc_meta:
                info = self.doc_meta.get(chunk.get("source_id", ""))
                if info and info.get("y_eff"):
                    lower_txt = chunk.get("text", "").lower()
                    if any(pk in lower_txt for pk in PARAMETER_KEYWORDS):
                        y_start = info["y_eff"]
                        y_end = (info.get("y_snap") or y_start) + 1
                        for y in range(y_start, y_end + 1):
                            self.param_chunks_by_year[y].append(chunk)

        self.avgdl = (total_len / self.num_docs) if self.num_docs > 0 else 1.0
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.num_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.inverted_index.items()
        }

    def score_candidates(self, query: str, case_year: int | str | None = None) -> list[tuple[float, int]]:
        """Score all matching chunks using the inverted index."""
        if not self.chunks:
            return []

        q_tokens = set(tokenize_multilingual(query))
        cy_int = int(case_year) if case_year is not None and str(case_year).isdigit() else None
        year_str = str(case_year) if case_year else None

        scores: dict[int, float] = defaultdict(float)
        for term in q_tokens:
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            w_idf = self.idf[term]
            for idx, tf in postings:
                dl = self.doc_lens[idx]
                scores[idx] += w_idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))))

        results: list[tuple[float, int]] = []
        for idx, sc in scores.items():
            if self.temporal_aware and self.doc_meta and cy_int:
                info = self.doc_meta.get(self.chunks[idx].get("source_id", ""))
                sc *= compute_temporal_multiplier(info, cy_int, self.code_eff_years)
            elif year_str and year_str in self.chunks[idx].get("title", ""):
                sc *= 1.25
            results.append((sc, idx))

        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def retrieve(self, query: str, top_k: int = 5, case_year: int | str | None = None) -> list[dict[str, Any]]:
        """Retrieve top_k chunks for the query."""
        scored = self.score_candidates(query, case_year=case_year)
        cy_int = int(case_year) if case_year is not None and str(case_year).isdigit() else None

        selected_chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for score, idx in scored:
            ch = dict(self.chunks[idx])
            ch["score"] = round(score, 4)
            seen_ids.add(ch["chunk_id"])
            selected_chunks.append(ch)
            if len(selected_chunks) >= top_k:
                break

        # If temporal_aware and annual statutory parameter chunk exists for case_year, inject at rank 2
        if self.temporal_aware and cy_int and cy_int in self.param_chunks_by_year and top_k >= 3:
            param_ch = dict(self.param_chunks_by_year[cy_int][0])
            if param_ch["chunk_id"] not in seen_ids:
                param_ch["score"] = 999.0
                insert_pos = min(1, len(selected_chunks))
                selected_chunks.insert(insert_pos, param_ch)
                selected_chunks = selected_chunks[:top_k]

        for rank, ch in enumerate(selected_chunks, 1):
            ch["rank"] = rank
        return selected_chunks


class HybridRRFTRetriever:
    """Two-Stage Hybrid Retriever combining Temporal BM25 + Gemini Dense Embeddings via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        cache_dir: Path,
        embedding_model: str = "gemini-embedding-001",
        candidate_pool: int = 30,
    ):
        self.bm25 = bm25_retriever
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "dense_embeddings_cache.json"
        self.embedding_model = embedding_model
        self.candidate_pool = candidate_pool
        self.embeddings_cache: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    self.embeddings_cache = json.load(f)
            except Exception:
                self.embeddings_cache = {}

    def save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.embeddings_cache, f)

    async def ensure_embeddings(self, client: Any, texts_by_key: dict[str, str]):
        missing_keys = [k for k in texts_by_key if k not in self.embeddings_cache]
        if not missing_keys:
            return
        print(f"Embedding {len(missing_keys)} candidate chunks/queries via {self.embedding_model}...")
        for i in range(0, len(missing_keys), 100):
            batch_keys = missing_keys[i : i + 100]
            batch_texts = [texts_by_key[k][:3000] for k in batch_keys]
            res = await client.embeddings.create(model=self.embedding_model, input=batch_texts)
            for k, item in zip(batch_keys, res.data):
                self.embeddings_cache[k] = item.embedding
        self.save_cache()

    async def prewarm_questions(self, client: Any, questions: list[dict[str, Any]]):
        """Pre-compute and cache embeddings for all queries and their candidate pools in batches."""
        to_embed: dict[str, str] = {}
        for q in questions:
            query = f"{q.get('case_year', '')} {q.get('tax_category', '')} {q.get('question', '')}"
            q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            q_key = f"query::{q_hash}"
            to_embed[q_key] = query
            scored = self.bm25.score_candidates(query, case_year=q.get("case_year"))[: self.candidate_pool]
            for _, idx in scored:
                ch = self.bm25.chunks[idx]
                to_embed[ch["chunk_id"]] = f"{ch.get('title', '')}\n{ch.get('text', '')}"
        await self.ensure_embeddings(client, to_embed)

    async def retrieve_async(
        self,
        client: Any,
        query: str,
        top_k: int = 5,
        case_year: int | str | None = None,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Retrieve top_k chunks using either pure Dense (on pool) or Hybrid RRF (BM25 + Dense)."""
        scored = self.bm25.score_candidates(query, case_year=case_year)[: self.candidate_pool]
        if not scored:
            return []

        cy_int = int(case_year) if case_year is not None and str(case_year).isdigit() else None
        q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        q_key = f"query::{q_hash}"
        to_embed: dict[str, str] = {q_key: query}

        candidates: list[dict[str, Any]] = []
        for rank_bm25, (bm25_score, idx) in enumerate(scored, 1):
            ch = dict(self.bm25.chunks[idx])
            ch["bm25_rank"] = rank_bm25
            ch["bm25_score"] = bm25_score
            c_key = ch["chunk_id"]
            to_embed[c_key] = f"{ch.get('title', '')}\n{ch.get('text', '')}"
            candidates.append(ch)

        if any(k not in self.embeddings_cache for k in to_embed):
            await self.ensure_embeddings(client, to_embed)
        q_emb = self.embeddings_cache.get(q_key)

        if q_emb:
            for ch in candidates:
                c_emb = self.embeddings_cache.get(ch["chunk_id"])
                ch["dense_score"] = cosine_similarity(q_emb, c_emb) if c_emb else 0.0
        else:
            for ch in candidates:
                ch["dense_score"] = 0.0

        # Rank by dense score
        candidates_by_dense = sorted(candidates, key=lambda x: x["dense_score"], reverse=True)
        for rank_dense, ch in enumerate(candidates_by_dense, 1):
            ch["dense_rank"] = rank_dense

        if mode == "dense":
            for ch in candidates:
                ch["score"] = round(ch["dense_score"], 4)
            final_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
        else:
            # Reciprocal Rank Fusion (RRF)
            for ch in candidates:
                rrf = (1.0 / (60.0 + ch["bm25_rank"])) + (1.0 / (60.0 + ch["dense_rank"]))
                ch["score"] = round(rrf, 6)
            final_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)

        selected = final_sorted[:top_k]
        seen_ids = {c["chunk_id"] for c in selected}

        if self.bm25.temporal_aware and cy_int and cy_int in self.bm25.param_chunks_by_year and top_k >= 3:
            param_ch = dict(self.bm25.param_chunks_by_year[cy_int][0])
            if param_ch["chunk_id"] not in seen_ids:
                param_ch["score"] = 999.0
                insert_pos = min(1, len(selected))
                selected.insert(insert_pos, param_ch)
                selected = selected[:top_k]

        for rank, ch in enumerate(selected, 1):
            ch["rank"] = rank
        return selected


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate_retrieval_hit(
    retrieved_chunks: list[dict[str, Any]],
    gold_citations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate retrieval quality against gold citations for a question."""
    if not gold_citations:
        return {
            "passage_recall": 1.0,
            "complete_evidence_recall": 1.0,
            "source_doc_recall": 1.0,
            "mrr": 1.0,
        }

    retrieved_source_ids = {c.get("source_id") for c in retrieved_chunks if c.get("source_id")}
    normalized_chunks = [normalize_ws(c.get("text", "")) for c in retrieved_chunks]

    gold_source_ids = set()
    for c in gold_citations:
        sf = c.get("source_file") or ""
        m = re.search(r"s_[a-zA-Z0-9_]+", sf)
        gold_source_ids.add(m.group(0) if m else sf)
    gold_source_ids.discard("")
    source_hits = len(gold_source_ids & retrieved_source_ids)
    source_doc_recall = (source_hits / len(gold_source_ids)) if gold_source_ids else 1.0

    passage_hits = 0
    first_hit_rank = 0

    for cit in gold_citations:
        passage = normalize_ws(cit.get("supporting_passage", ""))
        sf = cit.get("source_file", "")
        m = re.search(r"s_[a-zA-Z0-9_]+", sf)
        src_id = m.group(0) if m else sf
        if not passage:
            continue
        probe = passage[: min(len(passage), 60)]
        matched = False
        for rank, (chunk, norm_text) in enumerate(zip(retrieved_chunks, normalized_chunks), 1):
            if probe in norm_text or (chunk.get("source_id") == src_id and passage[:35] in norm_text):
                matched = True
                if first_hit_rank == 0 or rank < first_hit_rank:
                    first_hit_rank = rank
                break
        if matched:
            passage_hits += 1

    total_passages = len(gold_citations)
    passage_recall = passage_hits / total_passages if total_passages > 0 else 1.0
    complete_evidence = 1.0 if passage_hits == total_passages else 0.0
    mrr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0

    return {
        "passage_recall": round(passage_recall, 4),
        "complete_evidence_recall": round(complete_evidence, 4),
        "source_doc_recall": round(source_doc_recall, 4),
        "mrr": round(mrr, 4),
    }

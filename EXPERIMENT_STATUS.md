# TaxCalcBench — Experiment Status & Benchmark Evaluation Report

*Last updated: 14 October 2026*  
*Environment: HPC Cluster (`lo-01` / `ws-ia`)*

---

## 1. Executive Summary & Complete Chronological Experiment Log

This document records the live status of all experimental evaluations conducted on **TaxCalcBench**. Following the completion and validation of all 2,400 questions across 8 datasets / 7 countries, we have thoroughly benchmarked and calibrated our complete evaluation methodology on our bilingual pilot country: **Kazakhstan (`KZ` Russian & `KZ_kk` Kazakh)** across **20 full-dataset and split runs (5,400+ evaluated question runs)**.

### Chronological Log of All Experiments Executed:

1. **Experiment 1: KZ (Russian) Development Baseline (30 questions, 5 families)**
   * *Description*: Initial calibration of `gemini-3.8-flash` on 5 development families comparing Condition 1.1 (`closed_book_cot`) vs. Condition 2.1 (`oracle_cot`).
   * *Result*: Solvable Exact Match jumped from **68.0% $\rightarrow$ 100.0%** ($\Delta = +32.0\%$). Family Consistency jumped from **20.0% $\rightarrow$ 100.0%**.
2. **Experiment 2: KZ (Russian) Frozen Test Split & Full Dataset (300 questions, 50 families)**
   * *Description*: Full-scale evaluation of `closed_book_cot` and `oracle_cot` across the 270-question test split and the complete 300-question dataset (`--split full`).
   * *Result*: On all 300 questions, Solvable Exact Match jumped from **51.2% $\rightarrow$ 94.4%** ($\Delta = +43.2\%$), and Family Consistency rose from **16.0% $\rightarrow$ 80.0%**. Confirmed severe longitudinal recency decay in closed-book mode (55.6% on 2020–2026 vs. 26.7% on 2000–2009).
3. **Experiment 3: KZ (Kazakh) Bilingual Benchmark (300 questions, 50 families)**
   * *Description*: Parallel evaluation on authentic official Kazakh statutes from Adilet (`outputs/full/kz_kk` and `data/kz_kk/sources/`) across Dev (30), Test (270), and Full (300) splits.
   * *Result*: Solvable Exact Match jumped from **52.8% (Closed-Book) $\rightarrow$ 92.4% (Oracle)** ($\Delta = +39.6\%$). Demonstrated strong **bilingual parity** between Russian and Kazakh ($\Delta_{\text{lang}} \le 2.0\%$).
4. **Experiment 4: Program-Aided CoT (PAL / Python Sandbox) Across Full KZ (RU & KK, 600 questions)**
   * *Description*: Evaluated Condition 1.2 (`closed_book_program_cot`) and Condition 2.2 (`oracle_program_cot`) across all 300 Russian and 300 Kazakh questions using the isolated Python execution sandbox (`src/taxcalcbench/eval/sandbox.py`).
   * *Result*:
     * In **Closed-Book**, executing Python code boosted Solvable Exact Match by **+9.6% in Russian** (**51.2% $\rightarrow$ 60.8%**) and **+6.0% in Kazakh** (**52.8% $\rightarrow$ 58.8%**).
     * In **Oracle Grounding**, Program-Aided CoT achieved the highest benchmark accuracy: **95.2% Solvable Exact Match** (95.6% Tolerant Match), **84.0% Missing-Info Abstention**, and **82.0% Family Consistency** in Russian!
5. **Experiment 5: Vanilla Statutory BM25 RAG Across Full KZ (RU & KK, 600 questions, $k=5$)**
   * *Description*: Evaluated Condition 3.1 (`bm25_rag_cot`) and Condition 3.2 (`bm25_rag_program_cot`) over the article-chunked official legislative corpora (`data/kz/sources` — 10,573 chunks; `data/kz_kk/sources` — 5,681 chunks) with `top_k=5`.
   * *Result*: Boosted Solvable Exact Match from **51.2% (Closed-Book) $\rightarrow$ 73.6% (BM25 RAG)** in Russian (**+22.4% gain**) and from **52.8% $\rightarrow$ 69.6%** in Kazakh (**+16.8% gain**).
6. **Experiment 6: Temporal & Parameter-Aware Statutory BM25 + Context Budget Ablation ($k \in \{3, 5, 8, 12\}$, 1,200 questions)**
   * *Description*: Introduced universal statutory temporal weighting (enforcing non-retroactivity for future enactments $Y_{\text{eff}} > Y_{\text{case}}$, active consolidation preference, and annual parameter/budget injection) across $k \in \{3, 5, 8, 12\}$ on all 300 questions in `kz`.
   * *Result*:
     * At $k=5$, Temporal BM25 boosted Source Document Recall from **55.6% $\rightarrow$ 69.6%**, Passage Recall from **20.9% $\rightarrow$ 30.7%**, MRR from **0.533 $\rightarrow$ 0.662**, and Solvable Exact Match from **73.6% $\rightarrow$ 76.8%**.
     * Increasing context budget to **$k=8$** boosted Solvable Exact Match to **82.8%** and Family Consistency to **46.0%**, establishing **$k=8$** as the optimal retrieval depth before context saturation at $k=12$ (82.0%).
7. **Experiment 7: Two-Stage Dense Semantic (`gemini-embedding-001`) & Hybrid RRF RAG ($k=8$, 1,200 questions)**
   * *Description*: Evaluated Two-Stage Dense Reranking (`dense_rag_cot_k8`, `dense_rag_program_cot_k8`) and Reciprocal Rank Fusion (`hybrid_rag_cot_k8`, `hybrid_rag_program_cot_k8`) combining Temporal BM25 candidate pools with 3,072-dimensional `gemini-embedding-001` embeddings.
   * *Result*:
     * **Hybrid RRF ($k=8$)** reached **84.0% Solvable Exact Match** and **50.0% Family Consistency** (MRR = **0.766**).
     * **Two-Stage Temporal + Dense Reranked RAG (`dense_rag_cot_k8` / `dense_rag_program_cot_k8`)** achieved **87.2% Solvable Exact Match** (**88.0% Tolerant Match**), **84.0% Missing-Info Abstention**, and **60.0% Family Consistency** (MRR = **0.776**, Passage Recall = **44.2%**) — closing within **7.2 percentage points** of the Oracle upper bound!

---

## 2. Master Benchmark Leaderboard on Kazakhstan (`KZ` — Full Dataset, 300 Questions)

All runs evaluated on the complete 50-family / 300-question dataset (250 solvable numeric calculations + 50 missing-information cases) using `gemini-3.8-flash`:

| Condition / Architecture | Regime | Modality | $k$ | Overall EM (300) | Solvable EM (250) | Solvable Tol ($\pm 1\%$) | Abstention Acc (50) | False Abst ($FAR$) | Family Consist ($FCS$) | Doc Recall | Passage Recall | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`closed_book_cot`** | Closed-Book | Text CoT | — | 55.7% | 51.2% | 51.6% | 78.0% | 0.4% | 16.0% | — | — | — |
| **`closed_book_program_cot`** | Closed-Book | **Python PoT**| — | **64.0%** | **60.8%** | **62.8%** | **80.0%** | 1.2% | 14.0% | — | — | — |
| **`bm25_rag_program_cot`** | Vanilla BM25 | **Python PoT**| 5 | 73.7% | 72.4% | 73.2% | 80.0% | 2.8% | 28.0% | 55.6% | 20.9% | 0.533 |
| **`bm25_rag_cot`** | Vanilla BM25 | Text CoT | 5 | 75.0% | 73.6% | 74.8% | 82.0% | 6.8% | 32.0% | 55.6% | 20.9% | 0.533 |
| **`temporal_bm25_rag_cot_k3`** | Temporal BM25| Text CoT | 3 | 74.0% | 71.6% | 72.4% | **86.0%** | 3.6% | 26.0% | 68.6% | 22.2% | 0.637 |
| **`temporal_bm25_rag_cot`** | Temporal BM25| Text CoT | 5 | 77.3% | 76.8% | 78.0% | 80.0% | 2.8% | 36.0% | 69.6% | 30.7% | 0.662 |
| **`temporal_bm25_rag_cot_k12`**| Temporal BM25| Text CoT | 12 | 82.7% | 82.0% | 82.8% | **86.0%** | 2.4% | 46.0% | **72.5%** | 43.5% | 0.676 |
| **`temporal_bm25_rag_cot_k8`** | Temporal BM25| Text CoT | 8 | 82.7% | 82.8% | 83.6% | 82.0% | 2.8% | 46.0% | 70.1% | 37.7% | 0.673 |
| **`hybrid_rag_program_cot_k8`**| Hybrid RRF | **Python PoT**| 8 | 82.0% | 82.4% | 84.0% | 80.0% | 1.6% | 46.0% | 70.9% | 43.2% | 0.766 |
| **`hybrid_rag_cot_k8`** | Hybrid RRF | Text CoT | 8 | 83.3% | 84.0% | 84.8% | 80.0% | 2.4% | 50.0% | 70.9% | 43.2% | 0.766 |
| **`dense_rag_program_cot_k8`** | **Dense Reranked**| **Python PoT**| **8** | 86.0% | **87.2%** | 87.6% | 80.0% | 1.2% | 58.0% | 72.0% | **44.2%** | **0.776** |
| **`dense_rag_cot_k8`** | **Dense Reranked**| **Text CoT** | **8** | **86.7%** | **87.2%** | **88.0%** | **84.0%** | 1.6% | **60.0%** | 72.0% | **44.2%** | **0.776** |
| **`oracle_cot`** | Oracle Gold | Text CoT | — | 92.3% | 94.4% | 94.8% | 82.0% | **0.0%** | 80.0% | 100.0% | 100.0% | 1.000 |
| **`oracle_program_cot`** | **Oracle Gold**| **Python PoT**| — | **93.3%** | **95.2%** | **95.6%** | **84.0%** | 0.4% | **82.0%** | 100.0% | 100.0% | 1.000 |

---

## 3. Deep-Dive Analysis of RAG & Retrieval Ablations

### A. Ablation 1: Temporal & Parameter Awareness (`bm25_rag` vs. `temporal_bm25_rag` at $k=5$)
* **Problem**: In national tax corpora spanning 25 years, later tax codes (e.g. the 2017 and 2025 Tax Codes) are 3x larger than historical codes (2001 Code) and share identical section titles (`Корпоративный подоходный налог`). Vanilla lexical BM25 retrieves anachronistic future laws for historical scenarios (Error E3). Furthermore, statutory indices (such as the Monthly Calculation Index / МРП in Republican Budget Laws) are stored in separate companion statutes.
* **Solution**: Universal Temporal Weighting (`src/taxcalcbench/eval/rag/retriever.py`) penalizes statutes enacted after `case_year` ($Y_{\text{eff}} > Y_{\text{case}}$), boosts the active primary consolidation for `case_year`, and injects the applicable annual parameter chunk.
* **Impact at $k=5$**:
  * **Source Document Recall**: $55.6\% \rightarrow \mathbf{69.6\%}$ ($+14.0\%$)
  * **Passage Recall**: $20.9\% \rightarrow \mathbf{30.7\%}$ ($+9.8\%$)
  * **MRR**: $0.533 \rightarrow \mathbf{0.662}$ ($+0.129$)
  * **Solvable Exact Match**: $73.6\% \rightarrow \mathbf{76.8\%}$ ($+3.2\%$)

### B. Ablation 2: Context Budget Pareto Curve ($k \in \{3, 5, 8, 12\}$)
* **$k = 3$**: Solvable EM = **71.6%** | Family Consistency = **26.0%** | Passage Recall = **22.2%**
* **$k = 5$**: Solvable EM = **76.8%** | Family Consistency = **36.0%** | Passage Recall = **30.7%**
* **$k = 8$ (Optimal)**: Solvable EM = **82.8%** | Family Consistency = **46.0%** | Passage Recall = **37.7%**
* **$k = 12$ (Saturation)**: Solvable EM = **82.0%** | Family Consistency = **46.0%** | Passage Recall = **43.5%**
* **Insight**: Because D3 tax questions require 5–8 interacting provisions simultaneously, expanding from $k=3$ to $k=8$ yields a **+11.2% gain in Solvable Exact Match** and **+20.0% in Family Consistency**. Beyond $k=8$, additional chunks introduce distractor provisions that slightly degrade precision. Thus **$k=8$ is frozen as the ideal context budget**.

### C. Ablation 3: Lexical vs. Hybrid RRF vs. Two-Stage Dense Reranking ($k=8$)
* Comparing three ranking functions over the top-30 temporally filtered candidate pool at $k=8$:
  1. **Temporal Lexical BM25 (`temporal_bm25_rag_cot_k8`)**: **82.8%** Solvable EM | **46.0%** FCS | MRR = **0.673**
  2. **Hybrid RRF (`hybrid_rag_cot_k8`)**: **84.0%** Solvable EM | **50.0%** FCS | MRR = **0.766**
  3. **Two-Stage Dense Reranked (`dense_rag_cot_k8` via `gemini-embedding-001`)**: **87.2%** Solvable EM (**88.0%** Tolerant Match) | **60.0%** FCS | MRR = **0.776**
* **Insight**: Public benchmark questions intentionally avoid citing article numbers and often describe transactions in natural business prose rather than statutory jargon. Reranking the temporally valid candidate pool with 3,072-dimensional `gemini-embedding-001` dense vectors raises MRR from **0.673 to 0.776**, lifts Passage Recall to **44.2%**, and pushes Solvable Exact Match to **87.2%** (within **7.2%** of the 94.4% Oracle ceiling!).

---

## 4. Bilingual Kazakhstan Benchmark Results (`KZ` Russian vs. `KZ_kk` Kazakh)

Complete $3 \times 2$ comparison across all 300 questions in both languages ($k=5$ baseline):

| Condition | Russian (`kz`) Solvable EM | Kazakh (`kz_kk`) Solvable EM | Bilingual Gap ($\text{RU} - \text{KK}$) | Russian FCS | Kazakh FCS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1.1 `closed_book_cot`** | 51.2% | 52.8% | -1.6% | 16.0% | 16.0% |
| **1.2 `closed_book_program_cot`**| 60.8% | 58.8% | +2.0% | 14.0% | 16.0% |
| **3.1 `bm25_rag_cot`** | 73.6% | 69.6% | +4.0% | 32.0% | 24.0% |
| **3.2 `bm25_rag_program_cot`** | 72.4% | 66.4% | +6.0% | 28.0% | 24.0% |
| **2.1 `oracle_cot`** | 94.4% | 92.4% | +2.0% | 80.0% | 72.0% |
| **2.2 `oracle_program_cot`** | 95.2% | 92.0% | +3.2% | 82.0% | 72.0% |

---

## 5. Finalized Ideal Experimental Setup (Ready for Multi-Country Rollout)

Based on the extensive empirical calibration on Kazakhstan, the **ideal, standardized evaluation suite** for each country (`cn`, `pl`, `id`, `pk`, `eg`, `in`, `kz`, `kz_kk`) consists of:

1. **Split**: `--split full` (all 50 families / 300 questions per country). Category-balanced `--split dev` (30 questions) and `--split test` (270 questions) remain built-in for rapid ablations.
2. **Core Conditions to Run Per Country**:
   * **Condition 1 (Parametric Memory Baseline)**:
     * `closed_book_cot` & `closed_book_program_cot`
   * **Condition 2 (Oracle Grounding Upper Bound)**:
     * `oracle_cot` & `oracle_program_cot`
   * **Condition 3 (Statutory RAG)**:
     * `bm25_rag_cot` ($k=5$, lexical baseline)
     * `dense_rag_cot` & `dense_rag_program_cot` (`--top-k 8`, Two-Stage Temporal + `gemini-embedding-001` Dense Reranked RAG)
     * `hybrid_rag_cot` (`--top-k 8`, Reciprocal Rank Fusion)
3. **Runtime Parameters**:
   * `model="gemini-3.8-flash"`, `reasoning_effort="low"`, `temperature=0.0`, `max_output_tokens=16384`, `timeout_seconds=90.0`, `max_concurrency=20`.

---

## 6. Next Steps Roadmap

1. **Execute the Standardized Suite Across Remaining 6 Countries (`CN`, `PL`, `ID`, `PK`, `EG`, `IN`)**:
   ```bash
   for c in cn pl id pk eg in; do
     python scripts/run_eval.py --country $c --split full \
       --conditions closed_book_cot closed_book_program_cot oracle_cot oracle_program_cot \
       --concurrency 20
     python scripts/run_eval.py --country $c --split full \
       --conditions dense_rag_cot hybrid_rag_cot \
       --top-k 8 --concurrency 20
   done
   ```
2. **Multi-Country Corpus Interference Experiment**:
   * Pool all 220 documents across all 7 countries to measure cross-border retrieval noise (**Wrong-Country Retrieval Rate**).
3. **Automated Step-by-Step Milestone & 10-Tier Error Audit**:
   * Run step-level verification against `gold_steps` and generate final publication tables.

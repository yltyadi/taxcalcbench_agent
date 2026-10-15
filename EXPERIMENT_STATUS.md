# TaxCalcBench — Experiment Status & Benchmark Evaluation Report

*Last updated: 15 October 2026*  
*Environment: HPC Cluster (`lo-01` / `ws-ia`)*

---

## 1. Executive Summary & Complete Chronological Experiment Log

This document serves as our **live experimental lab notebook** recording all baseline runs, proof-of-concept (PoC) retrieval experiments, and empirical findings on **TaxCalcBench** (while `EXPERIMENT_DESIGN.md` remains our clean, formal gold specification document).

Following the completion and validation of all 2,400 questions across 8 datasets / 7 countries, we have thoroughly benchmarked and calibrated our evaluation methodology on our bilingual pilot country: **Kazakhstan (`KZ` Russian & `KZ_kk` Kazakh)** across **20 full-dataset and split runs (5,400+ evaluated question runs)**.

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
5. **Experiment 5: Vanilla Statutory BM25 RAG Baseline Across Full KZ (RU & KK, 600 questions, $k=5$)**
   * *Description*: Evaluated standard off-the-shelf keyword retrieval (`bm25_rag_cot` and `bm25_rag_program_cot`) over the article-chunked official legislative corpora (`data/kz/sources` — 10,573 chunks; `data/kz_kk/sources` — 5,681 chunks) with `top_k=5`.
   * *Result*: Boosted Solvable Exact Match from **51.2% (Closed-Book) $\rightarrow$ 73.6% (Vanilla BM25 RAG)** in Russian (**+22.4% gain**) and from **52.8% $\rightarrow$ 69.6%** in Kazakh (**+16.8% gain**).
6. **Experiment 6 (PoC): Temporal & Parameter-Aware Statutory BM25 + Context Budget Ablation ($k \in \{3, 5, 8, 12\}$, 1,200 questions)**
   * *Description*: Proof-of-concept experiment testing whether filtering out anachronistic future laws ($Y_{\text{eff}} > Y_{\text{case}}$) and injecting the annual Republican Budget parameter chunk (MCI/МРП) improves BM25 across $k \in \{3, 5, 8, 12\}$.
   * *Result*:
     * At $k=5$, Temporal BM25 boosted Source Document Recall from **55.6% $\rightarrow$ 69.6%**, Passage Recall from **20.9% $\rightarrow$ 30.7%**, MRR from **0.533 $\rightarrow$ 0.662**, and Solvable Exact Match from **73.6% $\rightarrow$ 76.8%**.
     * Increasing context budget to **$k=8$** boosted Solvable Exact Match to **82.8%** and Family Consistency to **46.0%**, establishing **$k=8$** as the optimal retrieval depth before context saturation at $k=12$ (82.0%).
7. **Experiment 7 (PoC): Two-Stage Dense Semantic Reranking (`gemini-embedding-001`) & Hybrid RRF RAG ($k=8$, 1,200 questions)**
   * *Description*: Proof-of-concept experiment comparing three ways to rank the top-30 candidate chunks retrieved by Temporal BM25: (a) keeping the BM25 keyword order, (b) re-ordering purely by 3,072-dimensional `gemini-embedding-001` vector similarity (`dense_rag_cot_k8`), or (c) combining both ranks equally via Reciprocal Rank Fusion (`hybrid_rag_cot_k8`).
   * *Result*:
     * **Hybrid RRF ($k=8$)** reached **84.0% Solvable Exact Match** and **50.0% Family Consistency** (MRR = **0.766**).
     * **Two-Stage Temporal + Dense Reranked RAG (`dense_rag_cot_k8` / `dense_rag_program_cot_k8`)** achieved **87.2% Solvable Exact Match** (**88.0% Tolerant Match**), **84.0% Missing-Info Abstention**, and **60.0% Family Consistency** (MRR = **0.776**, Passage Recall = **44.2%**) — closing within **7.2 percentage points** of the Oracle upper bound!

---

## 2. Master Benchmark Leaderboard on Kazakhstan (`KZ` — Full Dataset, 300 Questions)

All runs evaluated on the complete 50-family / 300-question dataset (250 solvable numeric calculations + 50 missing-information cases) using `gemini-3.8-flash`:

| Condition / Architecture | Category | Modality | $k$ | Overall EM (300) | Solvable EM (250) | Solvable Tol ($\pm 1\%$) | Abstention Acc (50) | False Abst ($FAR$) | Family Consist ($FCS$) | Doc Recall | Passage Recall | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`closed_book_cot`** | Baseline (No Law) | Text CoT | — | 55.7% | 51.2% | 51.6% | 78.0% | 0.4% | 16.0% | — | — | — |
| **`closed_book_program_cot`** | Baseline (No Law) | **Python PoT**| — | **64.0%** | **60.8%** | **62.8%** | **80.0%** | 1.2% | 14.0% | — | — | — |
| **`bm25_rag_program_cot`** | Baseline (Vanilla BM25)| **Python PoT**| 5 | 73.7% | 72.4% | 73.2% | 80.0% | 2.8% | 28.0% | 55.6% | 20.9% | 0.533 |
| **`bm25_rag_cot`** | Baseline (Vanilla BM25)| Text CoT | 5 | 75.0% | 73.6% | 74.8% | 82.0% | 6.8% | 32.0% | 55.6% | 20.9% | 0.533 |
| **`temporal_bm25_rag_cot_k3`** | PoC (Temporal BM25)| Text CoT | 3 | 74.0% | 71.6% | 72.4% | **86.0%** | 3.6% | 26.0% | 68.6% | 22.2% | 0.637 |
| **`temporal_bm25_rag_cot`** | PoC (Temporal BM25)| Text CoT | 5 | 77.3% | 76.8% | 78.0% | 80.0% | 2.8% | 36.0% | 69.6% | 30.7% | 0.662 |
| **`temporal_bm25_rag_cot_k12`**| PoC (Temporal BM25)| Text CoT | 12 | 82.7% | 82.0% | 82.8% | **86.0%** | 2.4% | 46.0% | **72.5%** | 43.5% | 0.676 |
| **`temporal_bm25_rag_cot_k8`** | PoC (Temporal BM25)| Text CoT | 8 | 82.7% | 82.8% | 83.6% | 82.0% | 2.8% | 46.0% | 70.1% | 37.7% | 0.673 |
| **`hybrid_rag_program_cot_k8`**| PoC (Hybrid RRF) | **Python PoT**| 8 | 82.0% | 82.4% | 84.0% | 80.0% | 1.6% | 46.0% | 70.9% | 43.2% | 0.766 |
| **`hybrid_rag_cot_k8`** | PoC (Hybrid RRF) | Text CoT | 8 | 83.3% | 84.0% | 84.8% | 80.0% | 2.4% | 50.0% | 70.9% | 43.2% | 0.766 |
| **`dense_rag_program_cot_k8`** | **PoC (Dense Reranked)**| **Python PoT**| **8** | 86.0% | **87.2%** | 87.6% | 80.0% | 1.2% | 58.0% | 72.0% | **44.2%** | **0.776** |
| **`dense_rag_cot_k8`** | **PoC (Dense Reranked)**| **Text CoT** | **8** | **86.7%** | **87.2%** | **88.0%** | **84.0%** | 1.6% | **60.0%** | 72.0% | **44.2%** | **0.776** |
| **`oracle_cot`** | Upper Bound (Oracle) | Text CoT | — | 92.3% | 94.4% | 94.8% | 82.0% | **0.0%** | 80.0% | 100.0% | 100.0% | 1.000 |
| **`oracle_program_cot`** | **Upper Bound (Oracle)**| **Python PoT**| — | **93.3%** | **95.2%** | **95.6%** | **84.0%** | 0.4% | **82.0%** | 100.0% | 100.0% | 1.000 |

---

## 3. Plain-English Walkthrough: What Each Retrieval Experiment Actually Does

To make it 100% clear what each experiment in the table above did under the hood, imagine we have a concrete question:
> **Example Question**: *"In **2008**, a resident company in Kazakhstan had gross revenue of 50,000,000 KZT and donated 1,000,000 KZT to charity. What is its Corporate Income Tax?"*

In our Kazakhstan folder (`data/kz/sources/`), we have **10,573 article chunks** coming from **four different Tax Codes** (the 2001 Code, the 2008 Code, the 2017 Code, and the 2025 Code) plus separate **Annual Budget Laws**.

Here is what each of the 4 retrieval methods did on this exact question:

---

### Method 1: Vanilla Lexical BM25 (`bm25_rag_cot`, Solvable Accuracy = 73.6%)
* **What it is**: The standard, off-the-shelf keyword search algorithm (like Ctrl+F with word rarity weights).
* **What it does step-by-step**:
  1. It breaks the question into words (`"2008"`, `"компания"`, `"доход"`, `"благотворительность"`, `"корпоративный"`, `"налог"`).
  2. It counts how many times those exact words appear in each of the 10,573 chunks.
  3. It picks the top 5 chunks with the highest keyword score.
* **Why it gets stuck at 73.6% (Two Big Flaws)**:
  * **Flaw A (Time Blindness)**: The 2017 Tax Code and 2025 Tax Code are 3x longer and repeat the words *"корпоративный подоходный налог"* and *"благотворительность"* more compactly than the 2001 Tax Code. So Vanilla BM25 accidentally hands the LLM the **2017 or 2025 Tax Code** (where the corporate tax rate is 20% and charity cap is 4%) instead of the **2001 Tax Code** (where the 2008 rate was 30% and charity cap was 3%)!
  * **Flaw B (Missing the Index Table)**: Many tax rules say *"the tax exemption is 14 МРП (Monthly Calculation Index)"*. The value of 1 МРП in 2008 (`1,168 KZT`) is stored in a separate law (`"Law on the Republican Budget for 2008"`). Because the question never mentions the words *"Republican Budget"* or *"МРП"*, Vanilla BM25 never retrieves that law!

---

### Method 2 (PoC): Temporal Lexical BM25 (`temporal_bm25_rag_cot`, Solvable Accuracy = 76.8% at $k=5$, 82.8% at $k=8$)
* **What it is**: A proof-of-concept experiment where we took Vanilla BM25 and added two simple domain rules based on the year in the document's title:
  1. **Rule 1 (Time Filter / Non-Retroactivity)**: Look at the question's year (`case_year = 2008`) and look at the year in each law's title (`"Tax Code of 25 December 2017"`). Since a law written in 2017 **did not exist in 2008**, we multiply the score of future laws by `0.05` (pushing them to the bottom) and multiply the score of the active 2001 Tax Code by `2.5` (pushing it to the top!).
  2. **Rule 2 (Annual Budget Parameter Injection)**: Since the question is in `2008`, we automatically reserve 1 slot in the retrieved passages for the `"Law on the Republican Budget for 2008"` chunk that lists the 2008 МРП (`1,168 KZT`) and Minimum Wage!
* **Why Accuracy Jumped from 73.6% $\rightarrow$ 82.8%**:
  * Suddenly, the LLM stopped receiving anachronistic 2025 laws for 2008 questions (**Document Recall jumped from 55.6% to 70.1%**), and it always had the exact МРП index value for that year!
  * Increasing the number of retrieved chunks from $k=5$ to $k=8$ added another **+6.0%** (from 76.8% to 82.8%) because a complex tax problem needs ~6–8 articles at once (Gross Income article + Deductions article + Charity cap article + Tax Rate article + Budget МРП).

---

### Method 3 (PoC): Two-Stage Temporal + Dense Reranked (`dense_rag_cot_k8`, Solvable Accuracy = 87.2%)
* **What it is**: A two-stage pipeline combining **Temporal BM25 (Stage 1)** with **Google's `gemini-embedding-001` Neural Vector Embeddings (Stage 2)**.
* **Why Temporal BM25 alone still missed some passages**:
  * In our benchmark, questions are written in **realistic everyday business language** (and explicitly forbidden from naming article numbers). For example, a question might say *"the company wrote off an unpaid customer invoice after 3 years"*, whereas the Tax Code calls this *"Deduction of Doubtful Liabilities (Сомнительные обязательства)"*.
  * Because the words don't match (`"unpaid customer invoice"` $\neq$ `"сомнительные обязательства"`), BM25 ranks that article #22 instead of #1—so at $k=8$, it gets cut off!
* **What Two-Stage Dense Reranking does step-by-step**:
  1. **Stage 1 (Candidate Pool)**: Temporal BM25 quickly grabs the **top 30 chunks** from the correct time period (2008). That guarantees the right article is somewhere inside those 30 candidates, and keeps out wrong-year laws.
  2. **Stage 2 (Semantic Vector Reranking)**: We convert the question and those 30 candidate chunks into 3,072-dimensional semantic meaning vectors using `gemini-embedding-001` and compute their cosine similarity.
  3. Because embedding vectors understand *meaning* rather than just exact letters, `"unpaid customer invoice after 3 years"` matches `"сомнительные обязательства (doubtful liabilities)"` with high similarity, moving it from rank #22 up to rank #2!
  4. We hand the top 8 reranked chunks to the LLM.
* **Result**: **Passage Recall jumped from 37.7% $\rightarrow$ 44.2%**, **MRR jumped from 0.673 $\rightarrow$ 0.776**, **Family Consistency jumped from 46.0% $\rightarrow$ 60.0%**, and **Solvable Exact Match reached 87.2%**!

---

### Method 4 (PoC): Hybrid RRF (`hybrid_rag_cot_k8`, Solvable Accuracy = 84.0%)
* **What it is**: Instead of letting the Dense Vector score completely override the BM25 keyword order on those 30 candidates (as Method 3 did), **Reciprocal Rank Fusion (RRF)** averages their two rankings equally (50/50):
  $$\text{RRF Score} = \frac{1}{60 + \text{BM25 Rank}} + \frac{1}{60 + \text{Dense Vector Rank}}$$
* **For example**: If a chunk is ranked #24 by BM25 keywords and #2 by Dense Vectors, RRF compromises and puts it around #9.
* **Result**: It scored **84.0% Solvable EM** and **50.0% Family Consistency** (better than pure Temporal BM25's 82.8%, but **-3.2% Solvable EM** and **-10.0% Family Consistency** worse than Two-Stage Dense Reranking's **87.2%** and **60.0%**).

---

### Deep-Dive Analysis: Why Did Hybrid RRF (`84.0%`) Perform Worse Than Two-Stage Dense Reranking (`87.2%`)?

This is one of our most important empirical findings, and it explains why a **step-by-step, stage-wise architecture (like Two-Stage Dense Reranking or FINCARDS)** outperforms standard 50/50 Hybrid RRF on legal tax benchmarks:

1. **The "Legal Vocabulary Mismatch" Problem in Tax Questions**:
   * By design, TaxCalcBench questions are written in **natural business prose** and are strictly forbidden from naming article numbers or copying statutory jargon verbatim.
   * For example:
     * Question prose: *"wrote off an unpaid customer invoice after 3 years"* $\leftrightarrow$ Statute jargon: *"вычет по сомнительным обязательствам (deduction of doubtful liabilities)"*.
     * Question prose: *"paid for employee lunches at the factory"* $\leftrightarrow$ Statute jargon: *"доход работника в натуральной форме (employee income in kind)"*.
2. **What Happens in Stage 1 (Candidate Pool Selection)**:
   * In Stage 1, `Temporal BM25` already accomplished everything keyword search is good for: it used broad domain keywords (`"корпоративный налог"`, `"доход"`, `"работник"`, `"2008"`) and temporal filtering ($Y_{\text{eff}} \le Y_{\text{case}}$) to narrow 10,573 chunks down to **30 candidate articles from the right Tax Code and right year**.
3. **Why 50/50 RRF Averaging Hurts in Stage 2 (Dilution by Generic Articles)**:
   * Inside that pool of 30 candidates:
     * A **generic introductory article** (e.g. `"Article 222: Taxpayers of Corporate Income Tax"` or `"Article 224: Concept of Taxable Income"`) repeats the words `"корпоративный подоходный налог"` 15 times $\rightarrow$ **BM25 Rank = #1**, **Dense Semantic Rank = #18**.
     * The **specific calculation article actually needed** (e.g. `"Article 248: Doubtful Liabilities"`) only mentions `"налог"` once and uses specialized terminology $\rightarrow$ **BM25 Rank = #24**, **Dense Semantic Rank = #2**.
   * When **Hybrid RRF** averages BM25 Rank and Dense Rank 50/50:
     * The generic introductory article (#1 + #18) gets boosted into the top 5!
     * The specific calculation article (#24 + #2) gets dragged down to **Rank #9 or #10** — **falling just outside the top-$k=8$ cutoff!**
4. **Why Two-Stage Dense Reranking (`dense_rag_cot_k8`, `87.2%`) Wins**:
   * Instead of mixing noisy keyword counts into the final ranking, **Two-Stage Dense Reranking uses a clean sequential division of labor** (the exact same philosophy as a step-by-step FINCARDS pipeline):
     * **Step 1 (Hard Structural & Temporal Filter)**: Use `Temporal BM25` + non-retroactivity ($Y_{\text{eff}} \le Y_{\text{case}}$) strictly as a **candidate gatekeeper** to get 30 articles from the right jurisdiction, year, and tax type.
     * **Step 2 (Pure Semantic Precision Reranking)**: Once the 30 candidates are already guaranteed to be from the right law and year, throw away the raw BM25 keyword counts and rank the 30 candidates **100% by deep semantic meaning (`gemini-embedding-001`)**!
   * **Quantitative Proof of Superiority**:
     * **Passage Recall@8**: `37.7%` (Temporal BM25) $\rightarrow$ `43.2%` (Hybrid RRF) $\rightarrow$ **`44.2%`** (Two-Stage Dense Reranked)
     * **Mean Reciprocal Rank (MRR)**: `0.673` (Temporal BM25) $\rightarrow$ `0.766` (Hybrid RRF) $\rightarrow$ **`0.776`** (Two-Stage Dense Reranked)
     * **Solvable Exact Match**: `82.8%` (Temporal BM25) $\rightarrow$ `84.0%` (Hybrid RRF) $\rightarrow$ **`87.2%`** (Two-Stage Dense Reranked, **+3.2%** over Hybrid RRF)
     * **Family Consistency Score (FCS)**: `46.0%` (Temporal BM25) $\rightarrow$ `50.0%` (Hybrid RRF) $\rightarrow$ **`60.0%`** (Two-Stage Dense Reranked, **+10.0%** jump over Hybrid RRF!)

---

## 4. How Our Working PoC Architecture & Your Two New Ideas Fit Together

We keep **all** of our working methods (`Temporal BM25`, `Hybrid RRF`, and `Two-Stage Temporal + Dense Reranked` at **87.2%**) as permanent, reproducible methods in our evaluation suite, and we can compare or combine them with two complementary extensions:

### A. Our Working Method (`dense_rag_cot_k8` — 87.2% EM, 60.0% FCS) vs. Full FINCARDS-Style Reranking
* Our **Two-Stage Temporal + Dense Reranked (`dense_rag_cot_k8`)** already implements the core step-by-step principle:
  1. Filter by temporal validity ($Y_{\text{eff}} \le Y_{\text{case}}$) and broad lexical domain.
  2. Inject the active annual parameter chunk (МРП / Minimum Wage).
  3. Rerank candidates purely by semantic similarity (`gemini-embedding-001`).
* **Testing FINCARDS-style Structured Cards Alongside It**:
  * We can test whether adding explicit structured card headers (`[Jurisdiction, Period, OECD Category, Taxpayer Entity, Parameters]`) on top of (or compared against) our Two-Stage Temporal + Dense Reranker pushes accuracy even closer to the 95.2% Oracle ceiling—especially in the **Multi-Country Corpus** setting where 7 countries' laws are mixed together!
  * Both methods will be kept and compared head-to-head so the empirical numbers decide which variant is strongest.

### B. Dual-Layer Thinking-Trace RAG (Agentic Dataset $\rightarrow$ Human Expert Test Set)
* Even with **87.2%** accuracy from our Two-Stage Dense Reranker (and **95.2%** from Oracle), the remaining errors happen when the LLM struggles to order complex 6-step tax deductions properly.
* Adding a **second retrieval layer** that fetches 1–3 worked step-by-step solutions (`gold_steps` / thinking traces) from our **agent-generated dataset** to guide the model when solving **human-expert-authored test questions** is completely orthogonal and can be stacked directly on top of our **87.2% Two-Stage Dense Reranker**!

---

## 5. Bilingual Kazakhstan Benchmark Results (`KZ` Russian vs. `KZ_kk` Kazakh)

Complete $3 \times 2$ baseline comparison across all 300 questions in both languages ($k=5$ baseline):

| Condition | Russian (`kz`) Solvable EM | Kazakh (`kz_kk`) Solvable EM | Bilingual Gap ($\text{RU} - \text{KK}$) | Russian FCS | Kazakh FCS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1.1 `closed_book_cot`** | 51.2% | 52.8% | -1.6% | 16.0% | 16.0% |
| **1.2 `closed_book_program_cot`**| 60.8% | 58.8% | +2.0% | 14.0% | 16.0% |
| **3.1 `bm25_rag_cot` (Vanilla)** | 73.6% | 69.6% | +4.0% | 32.0% | 24.0% |
| **3.2 `bm25_rag_program_cot`** | 72.4% | 66.4% | +6.0% | 28.0% | 24.0% |
| **2.1 `oracle_cot`** | 94.4% | 92.4% | +2.0% | 80.0% | 72.0% |
| **2.2 `oracle_program_cot`** | 95.2% | 92.0% | +3.2% | 82.0% | 72.0% |

---

## 6. Summary of Standard Retrieval Methods vs. PoC Variants

| Method Name in Logs | Type | How It Works in 1 Sentence |
| :--- | :---: | :--- |
| **`bm25_rag_cot`** | **Standard Baseline** | Pure keyword/stem matching across all 10,573 chunks with zero year filtering (`73.6%`). |
| **`temporal_bm25_rag_cot`** | **PoC Experiment** | Keyword matching + penalizing laws enacted after `case_year` + injecting the annual budget МРП chunk (`76.8%` at $k=5$, `82.8%` at $k=8$). |
| **`hybrid_rag_cot_k8`** | **PoC Experiment** | Takes the top-30 chunks from Temporal BM25 and averages their BM25 keyword rank with their `gemini-embedding-001` vector rank (`84.0%`). |
| **`dense_rag_cot_k8`** | **PoC Experiment** | Takes the top-30 chunks from Temporal BM25 and re-orders them purely by `gemini-embedding-001` semantic vector similarity (`87.2%`). |
| **FINCARDS Reranking** | **Planned Phase 2** | Principled structured card reranking over `[Jurisdiction, Period, Category, Entity]` metadata + semantics. |
| **Thinking-Trace RAG** | **Planned Phase 2** | Dual-layer retrieval fetching both official statutes (Layer 1) and step-by-step worked solutions from the agentic dataset (Layer 2). |

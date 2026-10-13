# TaxCalcBench — Experiment Status & Pilot Evaluation Report

*Last updated: 13 October 2026*  
*Environment: HPC Cluster (`lo-01` / `ws-ia`)*

---

## 1. Executive Summary

This report documents the live status of the **TaxCalcBench** experimental evaluation phase. Following the successful generation and verification of all national tax calculation benchmark datasets (2,400 questions across 8 datasets / 7 countries), we executed the **first version of empirical experiments** on the foundational pilot dataset: **Kazakhstan (`KZ` - Russian)** across both the **Development Set (30 questions)** and the **Full Evaluation Test Set (270 questions)**.

### Key Milestones Achieved:
* **Evaluation Harness Deployed**: Built a dedicated, modular evaluation engine (`src/taxcalcbench/eval/` and `scripts/run_eval.py`) supporting asynchronous execution with Gemini API, incremental disk checkpointing, LaTeX boxed answer extraction, and multi-tier metric computation.
* **Pilot Experiments Completed on Kazakhstan (Russian)**:
  * **Condition 1.1: Closed-Book CoT** (Parametric memory only, 0 context) — **300 / 300 questions evaluated**.
  * **Condition 2.1: Oracle Grounding CoT** (Gold statutory passages injected from `citations.json`) — **300 / 300 questions evaluated**.
* **Empirical Validation of Core Hypotheses on the 270-Question Frozen Test Set**:
  * **Closed-Book Solvable Exact Match is 47.6%**, while **Oracle Grounding reaches 93.8%** ($\Delta = \mathbf{+46.2\%}$ percentage points). Providing official statutory text nearly doubles the model's accuracy.
  * **Family Consistency Score (FCS)** jumps from **13.3%** in closed-book to **77.8%** under oracle grounding ($\Delta = \mathbf{+64.5\%}$). Closed-book models fail to solve complete families consistently.
  * **Longitudinal Recency Bias Confirmed**: In closed-book mode, accuracy drops from **55.6%** on contemporary laws (`2020–2026`) down to **26.7%** on historical laws (`2000–2009`) ($\Delta_{\text{temp}} = \mathbf{28.9\%}$). Oracle grounding restores historical accuracy to **86.7%**.

---

## 2. Experimental Results: Full Evaluation Test Set (Kazakhstan — Russian)

Evaluation conducted on the **45 Frozen Test Families (270 questions)**:
* Model: `gemini-3.8-flash` (Temperature: 0.0, max tokens: 8,192).
* Partition: Families `A-F06` to `A-F50` (225 regular calculation cases + 45 missing-information cases).
* Checkpoints: `outputs/eval/kz/` (`closed_book_cot/`, `oracle_cot/`, and `comparison_test.json`).

### A. Headline Metric Summary (Test Split — 270 Questions)

| Metric | Condition 1.1: Closed-Book CoT | Condition 2.1: Oracle Grounding CoT | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Exact Match ($Acc_{EM}$)** | 52.2% (141 / 270) | **91.5%** (247 / 270) | **+39.3%** |
| **Solvable Exact Match** | 47.6% (107 / 225) | **93.8%** (211 / 225) | **+46.2%** |
| **Solvable Tolerant Match ($Acc_{tol}$)** | 48.0% (108 / 225) | **94.2%** (212 / 225) | **+46.2%** |
| **Missing Info Abstention ($Acc_{abs}$)** | 75.6% (34 / 45) | **80.0%** (36 / 45) | **+4.4%** |
| **False Abstention Rate ($FAR$)** | 0.4% (1 / 225) | **0.0%** (0 / 225) | -0.4% |
| **Family Consistency Score ($Acc_{fam}$)** | **13.3%** (6 / 45 families) | **77.8%** (35 / 45 families) | **+64.5%** |

---

### B. Breakdown by Reasoning Variant (Closed-Book vs. Oracle)

| Question Variant | Total Cases | Closed-Book Exact Match | Oracle Exact Match | Delta ($\Delta$) | Key Diagnostic Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `original` | 45 | 57.8% (26 / 45) | **97.8%** (44 / 45) | **+40.0%** | Base scenarios heavily improved by citation text. |
| `numbers` | 6 | 33.3% (2 / 6) | **100.0%** (6 / 6) | **+66.7%** | Numeric changes without legal guidance lead to parametric drift. |
| `temporal` | 45 | 26.7% (12 / 45) | **86.7%** (39 / 45) | **+60.0%** | **Severe closed-book collapse**: model projects wrong year's law. |
| `threshold` | 88 | 44.3% (39 / 88) | **93.2%** (82 / 88) | **+48.9%** | Crossing caps/brackets breaks closed-book reasoning. |
| `eligibility` | 44 | 65.9% (29 / 44) | **97.7%** (43 / 44) | **+31.8%** | Qualifying exemptions strongly assisted by statutory text. |
| `missing_information`| 45 | 75.6% (34 / 45) | **80.0%** (36 / 45) | **+4.4%** | Good baseline humility; rarely fabricates missing inputs. |

---

### C. Breakdown by Historical Era (Longitudinal Recency Bias)

| Time Period | Total Solvable Cases | Closed-Book Exact Match | Oracle Exact Match | Closed-Book Decay |
| :--- | :---: | :---: | :---: | :---: |
| **Contemporary (2020–2026)** | 135 solvable | **55.6%** (75 / 135) | **97.8%** (132 / 135) | Baseline |
| **Intermediate (2010–2019)** | 45 solvable | **44.4%** (20 / 45) | **88.9%** (40 / 45) | -11.2% |
| **Historical (2000–2009)** | 45 solvable | **26.7%** (12 / 45) | **86.7%** (39 / 45) | **-28.9%** |

**Empirical Finding**: Closed-book parametric memory exhibits massive recency bias ($\Delta_{\text{temp}} = 28.9\%$). When calculating taxes under historical 2000s codes, the model repeatedly applies modern rates or deductions. Injecting the historical statute (Oracle condition) eliminates this failure mode, boosting 2000s accuracy from **26.7% to 86.7%**.

---

## 3. Experimental Results: Development Calibration Split (30 Questions)

Evaluation conducted on **5 Development Families (`A-F01` to `A-F05`, 30 questions)** used for setup validation:

| Metric | Condition 1.1: Closed-Book CoT | Condition 2.1: Oracle Grounding CoT | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Exact Match ($Acc_{EM}$)** | 73.3% (22 / 30) | **100.0%** (30 / 30) | **+26.7%** |
| **Solvable Exact Match** | 68.0% (17 / 25) | **100.0%** (25 / 25) | **+32.0%** |
| **Solvable Tolerant Match ($Acc_{tol}$)** | 68.0% (17 / 25) | **100.0%** (25 / 25) | **+32.0%** |
| **Missing Info Abstention ($Acc_{abs}$)** | **100.0%** (5 / 5) | **100.0%** (5 / 5) | 0.0% |
| **False Abstention Rate ($FAR$)** | **0.0%** (0 / 25) | **0.0%** (0 / 25) | 0.0% |
| **Family Consistency Score ($Acc_{fam}$)** | **20.0%** (1 / 5 families) | **100.0%** (5 / 5 families) | **+80.0%** |

---

## 4. Understanding the Experimental Conditions: Concepts & Terminology

### A. What Does "Closed-Book" vs. "Oracle" Mean? (Context Dimension)
* **Closed-Book (Condition 1)**:
  * The model receives **only the question prompt** (facts, dates, jurisdiction, and requested endpoint).
  * **Zero statutory text or documents are provided in context.**
  * The model must rely entirely on whatever legal knowledge was absorbed during pre-training ("parametric memory").
* **Oracle Grounding (Condition 2)**:
  * The model receives the question prompt **plus the exact gold statutory passages** extracted from `citations.json` (official law title, article number, and verbatim supporting quote).
  * The model does not need to search or retrieve anything. This establishes the **empirical upper bound** for pure legal comprehension and mathematical reasoning.
* *(Contrast with RAG - Condition 3)*:
  * In **RAG**, the model is not given the gold answer passages directly; instead, a search algorithm (BM25 or dense vector search) searches the full legal corpus to find candidate passages.

### B. What Does "Chain-of-Thought (CoT)" Mean? (Reasoning Dimension)
* **Direct Prompting**: Asking the model: *"What is the tax? Answer with just the number."*
  * The model is forced to predict the final number immediately in its very first generated token without scratchpad space. For complex multi-tier tax calculations, direct prompting almost always fails.
* **Chain-of-Thought (CoT)**: Prompting the model: *"Think step by step before answering. Lay out your legal reasoning, intermediate deductions, and statutory calculations, and write your final answer in `\boxed{value}`."*
  * CoT gives the model working memory tokens to break down the problem into sequential steps (e.g. determining residence $\rightarrow$ applying statutory deductions $\rightarrow$ computing progressive brackets $\rightarrow$ applying credits $\rightarrow$ final rounding).
* **Program-Aided CoT (PoT / PAL)**:
  * Instructing the model to express its reasoning as an executable Python script (`ans = ...`), which is executed in an isolated Python interpreter to perform exact decimal arithmetic.

---

## 5. Bilingual Kazakhstan Experiment Design (Russian vs. Kazakh)

Kazakhstan is our unique testbed because laws are promulgated with identical legal force in both **Russian** and **Kazakh**, and the repository contains matched official datasets for both languages:
* `outputs/full/kz/`: 300 questions, 2,578 citations in Russian (`data/kz/sources/`).
* `outputs/full/kz_kk/`: 300 questions, 2,578 citations in Kazakh (`data/kz_kk/sources/`).

### How We Experiment with the Dual-Language Setup:

#### 1. Matched Parallel Evaluation (Linguistic Parity / Disparity Benchmark)
* **Goal**: Measure whether frontier LLMs achieve equal tax calculation reasoning when presented with identical scenarios in Russian vs. Kazakh.
* **Setup**:
  * Run Condition 1 (Closed-Book) and Condition 2 (Oracle) on `outputs/full/kz` (Russian) and `outputs/full/kz_kk` (Kazakh) across the exact same 300 questions.
  * Compute the **Bilingual Performance Gap**:
    $$\Delta_{\text{lang}} = Acc_{EM}(\text{Russian}) - Acc_{EM}(\text{Kazakh})$$
  * Evaluate whether Kazakh performance suffers from lower token representation, morphological complexity, or training data imbalance.

#### 2. Cross-Lingual Statutory Retrieval (Cross-Language RAG)
* **Scenario A (Native Search)**: Kazakh question $\rightarrow$ retrieve from Kazakh corpus (`data/kz_kk/sources/`).
* **Scenario B (Cross-Lingual Search)**: Kazakh question $\rightarrow$ retrieve from Russian corpus (`data/kz/sources/`) or vice versa using multilingual dense embeddings (`BGE-M3`).
* **Goal**: Determine whether cross-lingual retrieval bridges statutory gaps in multilingual jurisdictions.

#### 3. Translation-Assisted vs. Native Reasoning
* Compare:
  1. **Direct Native Reasoning**: Solving Kazakh tax questions directly in Kazakh.
  2. **Translate-then-Solve**: Machine translating the prompt into Russian or English, reasoning, and mapping back the answer.

---

## 6. Overall Benchmark Datasets Inventory (All 7 Countries)

| Country / Dataset | Language | Status | Validated Families | Questions | Citations | Official Corpus |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kazakhstan (`KZ` - RU)** | Russian (`ru`) | **complete** | **50 / 50** | **300 / 300** | 2,578 | 53 files (`data/kz/sources/`) |
| **Kazakhstan (`KZ` - KK)** | Kazakh (`kk`) | **complete** | **50 / 50** | **300 / 300** | 2,578 | 15 files (`data/kz_kk/sources/`) |
| **China (`CN`)** | Chinese (`zh`) | **complete** | **50 / 50** | **300 / 300** | 2,807 | 32 files (`data/cn/sources/`) |
| **Poland (`PL`)** | Polish (`pl`) | **complete** | **50 / 50** | **300 / 300** | 2,067 | 33 files (`data/pl/sources/`) |
| **Indonesia (`ID`)** | Indonesian (`id`) | **complete** | **50 / 50** | **300 / 300** | 2,179 | 27 files (`data/id/sources/`) |
| **Pakistan (`PK`)** | English (`en`) | **complete** | **50 / 50** | **300 / 300** | 1,549 | 19 files (`data/pk/sources/`) |
| **Egypt (`EG`)** | Arabic (`ar`) | **complete** | **50 / 50** | **300 / 300** | 1,326 | 22 files (`data/eg/sources/`) |
| **India (`IN`)** | English (`en`) | **complete** | **50 / 50** | **300 / 300** | 1,801 | 19 files (`data/in/sources/`) |
| **GLOBAL TOTAL** | **6 languages** | **complete** | **400 / 400** | **2,400 / 2,400** | **16,885** | **220 files** |

---

## 7. Next Steps Roadmap

1. **Execute Kazakh Bilingual Evaluation (`KZ` - Kazakh)**:
   * Run Closed-Book CoT and Oracle CoT on `kz_kk` Dev and Test sets:
     ```bash
     python scripts/run_eval.py --country kz_kk --split dev --conditions closed_book_cot oracle_cot
     python scripts/run_eval.py --country kz_kk --split test --conditions closed_book_cot oracle_cot
     ```
   * Quantify the empirical **Bilingual Performance Gap** ($\Delta_{\text{lang}}$).
2. **Implement Program-Aided CoT (Condition 1.2 & 2.2)**:
   * Add sandboxed Python code execution to isolate pure arithmetic calculation mistakes from legal statutory interpretation errors.
3. **Build BM25 & Dense RAG Index for Kazakhstan (Condition 3)**:
   * Index `data/kz/sources/text/` and `data/kz_kk/sources/text/` by statutory section boundaries.
   * Run multi-retriever RAG evaluations ($k \in \{3, 5, 8\}$).
4. **Scale to All Countries**:
   * Roll out the calibrated evaluation pipeline to `CN`, `PL`, `ID`, `PK`, `EG`, and `IN`.

# TaxCalcBench: Unified Experimental Evaluation Framework
## Assessing Large Language Models on Statutory Tax Calculation via Controlled Retrieval and Reasoning Regimes

---

## 1. Executive Summary & Strategic Approach

### 1.1 Context & Current Milestone
With the completion of the automated generation and validation pipeline (specified in `CONTEXT_WINDOW.md` and guided by `references/Expert_Guide_and_Taxonomy.pdf`), the **TaxCalcBench** dataset is complete and verified across all **7 target jurisdictions**:
- **7 Countries / Jurisdictions**: Kazakhstan (`KZ`), China (`CN`), Poland (`PL`), Indonesia (`ID`), Pakistan (`PK`), Egypt (`EG`), and India (`IN`).
- **350 Validated Families** (50 families per country).
- **2,100 Audited Questions** (300 questions per country: 250 regular solvable calculations + 50 derived missing-information cases).
- **14,307 Primary Statutory Citations** directly grounded in official legislation across three historical eras (`2020–2026`, `2010–2019`, `2000–2009`).

### 1.2 Unified Strategy: Pilot-First with Google Gemini via API
To ensure maximum explainability, rigorous scientific control, and rapid iteration during the planning and testing stages:
1. **Pilot Jurisdiction — Kazakhstan (`KZ`) First**:
   - Kazakhstan is established as the foundational pilot country. With full linguistic fluency in Russian and Kazakh available for qualitative evaluation, KZ provides an ideal testbed: **50 families, 300 questions, 2,578 citations**, and **53 official legislative documents** spanning four major Tax Code eras (2001, 2008, 2017, and 2025).
2. **Standardized Model Architecture — Google Gemini API**:
   - Testing will center on **Google Gemini models via API calls** (`gemini-3.8-flash` as the fast, efficient primary baseline, and `gemini-2.5-pro` for deep reasoning comparison). This leverages the pre-built, robust API wrapper in `src/taxcalcbench/experts.py` with automated exponential backoff and disk checkpointing.
3. **Rigorous Family-Preserving Dev/Test Partitioning**:
   - The pilot will split Kazakhstan's 50 families into **5 Development Families (30 questions)** for prompt calibration, chunking optimization, and parser verification, and **45 Evaluation Families (270 questions)** held completely frozen for benchmark reporting.
4. **Comprehensive Evaluation Matrix**:
   - Synthesizes all core evaluation regimes: **No Retrieval (Parametric)**, **Oracle Grounding (Gold Evidence)**, and **Multi-Retriever RAG (Lexical BM25, Dense, Hybrid, Reranked)**.
   - Evaluates two reasoning modalities across all conditions: **Standard Chain-of-Thought (CoT)** and **Program-Aided CoT (Python execution)**.
   - Evaluates both **Single-Country Corpus** and **Multi-Country Corpus** settings to measure cross-border interference.
   - Employs both **Strict Exact Match** and **Lenient / Tolerant Matching**, alongside complete-evidence retrieval metrics and epistemic abstention scoring.

---

## 2. Benchmark Inventory & Ground Truth Characteristics

### 2.1 Overall Benchmark Statistics

| Country | Code | Primary Language | Validated Families | Total Questions | Solvable (Regular) | Missing Info | Auditable Citations | Official Corpus Files |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kazakhstan (Pilot - RU)** | `KZ` | Russian (`ru`) | **50** | **300** | **250** | **50** | **2,578** | **53** |
| **Kazakhstan (Pilot - KK)** | `KZ` | Kazakh (`kk`) | **50** | **300** | **250** | **50** | **2,578** | **15** |
| **China** | `CN` | Chinese (`zh`) | 50 | 300 | 250 | 50 | 2,807 | 32 |
| **Poland** | `PL` | Polish (`pl`) | 50 | 300 | 250 | 50 | 2,067 | 33 |
| **Indonesia** | `ID` | Indonesian (`id`) | 50 | 300 | 250 | 50 | 2,179 | 27 |
| **Pakistan** | `PK` | English (`en`) | 50 | 300 | 250 | 50 | 1,549 | 19 |
| **Egypt** | `EG` | Arabic (`ar`) | 50 | 300 | 250 | 50 | 1,326 | 22 |
| **India** | `IN` | English (`en`) | 50 | 300 | 250 | 50 | 1,801 | 19 |
| **GLOBAL TOTAL** | **7** | **6 languages** | **350** | **2,400** | **2,000** | **400** | **16,885** | **220** |

### 2.2 Deep Dive: Kazakhstan Dual-Language Pilot (`outputs/full/kz/` & `outputs/full/kz_kk/`)
Kazakhstan serves as the foundational bilingual pilot. In Kazakhstan, all national statutes are promulgated with equal legal force in both Russian and Kazakh:
- **Russian Dataset (`outputs/full/kz/`)**: 300 questions, 2,578 citations grounded in official Russian texts (`data/kz/sources/`).
- **Kazakh Dataset (`outputs/full/kz_kk/`)**: 300 questions, 2,578 citations grounded in authentic, official Kazakh texts (`data/kz_kk/sources/`). Citations feature verbatim official Kazakh supporting passages verified via `_source_quote`, without synthetic LLM translation of statutory quotes.
- **Matched Bilingual Evaluation**: Enables testing whether frontier LLMs exhibit linguistic parity or disparity when reasoning through identical statutory fact patterns in Russian versus Kazakh.
- **Temporal Distribution (Per Language)**:
  - Contemporary (`2020–2026`): 200 questions (150 regular + 50 missing-information).
  - Intermediate (`2010–2019`): 50 regular questions.
  - Historical (`2000–2009`): 50 regular questions.
- **OECD Tax Category Coverage**:
  - `1000` (Income, profits and capital gains): 60 questions
  - `2000` (Social security contributions): 42 questions
  - `3000` (Payroll and workforce taxes): 42 questions
  - `4000` (Property taxes): 54 questions
  - `5000` (Goods and services taxes / VAT): 48 questions
  - `6000` (Other taxes / mineral extraction / vehicle): 54 questions
- **Reasoning Complexity**: 299 D3 cases (multi-step dependent calculations involving caps, exceptions, and interacting rules) and 1 D2 case.
- **Official Legislative Corpus**: 53 verified Russian statutory files in `data/kz/sources/` and 15 corresponding primary official Kazakh statutory files in `data/kz_kk/sources/`, covering:
  - Tax Code of 12 June 2001 No. 209-II
  - Tax Code of 10 December 2008 No. 99-IV
  - Tax Code of 25 December 2017 No. 120-VI
  - Tax Code of 18 July 2025 No. 214-VIII
  - Associated Ministerial Decrees, Social Insurance Acts, and Mandatory Pension Contribution Rules.

---

## 3. Development and Evaluation Partitioning (Kazakhstan Pilot)

To guarantee scientific validity and eliminate data contamination, partitioning occurs strictly at the **family level**. All 6 variants of each family remain clustered in the same split:

```
                       KAZAKHSTAN BENCHMARK: 50 FAMILIES (300 QUESTIONS)
                                               │
                   ┌───────────────────────────┴───────────────────────────┐
                   ▼                                                       ▼
         DEVELOPMENT SET (Dev)                                   EVALUATION SET (Test)
        • 5 Families (30 Questions)                             • 45 Families (270 Questions)
        • Families: A-F01 to A-F05                              • Families: A-F06 to A-F50
        • 25 Solvable + 5 Missing Info                          • 225 Solvable + 45 Missing Info
        • Calibration & Parameter Tuning                        • FROZEN Benchmark Evaluation
```

### 3.1 Universal Dataset Partitioning & Dynamic Category-Aware Balancing
To prevent categorical or temporal bias when sampling development subsets across diverse jurisdictions:
- **Default Full Evaluation (`--split full` / `--split all`)**: Evaluates all 50 families (300 questions) in their entirety without artificial partitioning.
- **Dynamic Category-Aware Development Split (`--split dev`)**: Rather than hardcoding fixed family IDs, the partitioner dynamically groups families by their authoritative OECD `tax_category` and deterministically selects 1 representative family per distinct tax category present in that jurisdiction's corpus.
- **Evaluation Split (`--split test`)**: Evaluates the frozen complement (all remaining families).
- **Explicit Subsets (`--families` / `--limit`)**: Permits running custom family subsets or limiting evaluation to the first $N$ questions for rapid verification.

In Kazakhstan, the category-balanced split selects families representing 5 distinct tax bases:
- **`A-F01`**: Corporate Income Tax (`1000`, Contemporary `2020–2026`).
- **`A-F02`**: Immovable Property & Land Tax (`4000`, Intermediate `2010–2019`).
- **`A-F03`**: Value Added Tax (`5000`, Historical `2000–2009`).
- **`A-F05`**: Social Tax & Payroll Levies (`3000`, Contemporary `2020–2026`).
- **`A-F06`**: Vehicle / Other Taxes (`6000`, Contemporary `2020–2026`).

### 3.2 Calibration Objectives on the Development Set
The Development Set will be used exclusively to test, calibrate, and freeze:
1. **Passage Chunking Granularity**: Legal article/section boundaries vs. fixed token windows (300 vs. 600 vs. 1,000 tokens).
2. **Retrieval Depth ($k$)**: Comparing top-$k \in \{3, 5, 8, 12\}$ to balance statutory recall against LLM context clutter.
3. **Prompt Phrasing & Output Formatting**: Standardizing LaTeX box extraction (`\boxed{result} [UNIT]`) and JSON response parsing.
4. **Python Sandbox Execution Rules**: Tuning timeouts (5s limit), pre-imported libraries (`decimal.Decimal`), and variable capture.
5. **Scoring Margins & Rounding**: Calibrating strict vs. lenient tolerances against ground-truth values.

*(Note: All live experimental logs, baseline runs, and proof-of-concept ablation results are tracked separately in `EXPERIMENT_STATUS.md`).*

---

## 4. The Experimental Matrix: Retrieval Regimes $\times$ Reasoning Modalities

The experimental design evaluates models across a structured **3 Retrieval Regimes $\times$ 2 Reasoning Modalities** matrix, plus two corpus environments:

```
┌──────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│                          │          REASONING MODALITY 1   │          REASONING MODALITY 2   │
│ RETRIEVAL REGIME         │       Chain-of-Thought (CoT)    │     Program-Aided CoT (Python)  │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Condition 1:             │ 1.1 Closed-Book CoT             │ 1.2 Closed-Book Program CoT     │
│ Closed-Book (No Retr.)   │ (Parametric memory only)        │ (Generates & executes Python)   │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Condition 2:             │ 2.1 Oracle CoT                  │ 2.2 Oracle Program CoT          │
│ Oracle Grounding         │ (Exact gold passages injected)  │ (Gold passages + Python runtime)│
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Condition 3:             │ 3.1 RAG with CoT                │ 3.2 RAG with Program CoT        │
│ Multi-Retriever RAG      │ (Retrieved top-k chunks)        │ (Retrieved chunks + Python)     │
└──────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

---

### Condition 1: Closed-Book Parametric Reasoning (No Retrieval)
- **Concept**: The model receives only the scenario facts, jurisdiction, and `{case_year}` with zero external statutory text.
- **Objectives**:
  - Measures the extent of parametric pre-training memorization of Kazakhstan's tax codes.
  - Tests whether models can solve multi-step statutory problems without document grounding.
- **Variants**:
  - **1.1 Closed-Book CoT**: Standard step-by-step reasoning prompt: `"Solve this tax calculation for Kazakhstan in year {case_year}. Think step by step before stating your final answer."`
  - **1.2 Closed-Book Program-Aided CoT**: Prompt instructs model: `"Write executable Python code using exact arithmetic to calculate the tax liability for Kazakhstan in year {case_year}."` The code is executed in an isolated sandbox.

---

### Condition 2: Oracle Statutory Grounding (Gold Evidence Upper Bound)
- **Concept**: The model receives the question accompanied by the exact gold statutory provisions extracted from `citations.json` (official law title, law number, article/section, and verbatim `supporting_passage`).
- **Objectives**:
  - Establishes the **empirical upper bound** for statutory interpretation and calculation.
  - Isolates legal comprehension and arithmetic from retrieval failure. If a model fails here, the limitation lies in statutory synthesis or mathematical logic, not in finding the law.
- **Variants**:
  - **2.1 Oracle CoT**: Gold statutory passages provided directly in the prompt context.
  - **2.2 Oracle Program-Aided CoT**: Gold passages provided; model writes and executes Python code to perform the calculation.

---

### Condition 3: Multi-Retriever RAG (Standard Baselines vs. Proposed Advanced Architectures)
- **Concept**: Models receive passages retrieved from the official statutory corpus (`data/kz/sources/`).
- **A. Standard Off-the-Shelf Baselines (Testing Existing Methods First)**:
  To benchmark existing retrieval methods fairly without custom heuristics, we evaluate four standard baselines:
  1. **Lexical Retriever (Vanilla BM25)**: Pure keyword/token frequency and inverse document frequency matching. Strong on exact statutory numbers, article numbers, and specialized legal terminology, but blind to semantic paraphrases and temporal version dates.
  2. **Dense Semantic Retriever**: Multilingual dense embeddings (`gemini-embedding-001` or `BGE-M3`). Effective at semantic paraphrasing and conceptual legal queries, but prone to blurring across historical versions or jurisdictions.
  3. **Hybrid Retriever (BM25 + Dense RRF)**: Reciprocal Rank Fusion (RRF) combining lexical and dense rankings:
     $$\text{RRF Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
  4. **Standard Cross-Encoder Reranker**: Top-20 candidates from hybrid retrieval re-ranked using an off-the-shelf reranker (`bge-reranker-large` / `cohere-rerank-v3`).
- **B. Proposed Advanced Research Directions (Phase 2 Methodological Contributions)**:
  Once standard baselines establish the retrieval bottleneck, we evaluate two novel domain-informed architectures:
  1. **Direction 1: FINCARDS-Style Structured Jurisdiction & Period-Aware Reranking**:
     - Rather than ad-hoc keyword weighting, each statutory chunk is indexed into a structured **TaxCard / LegalCard** schema: `[Jurisdiction, Law Title, Effective Period (rule_applies_from -> rule_applies_until), OECD Tax Category, Taxpayer Entity, Statutory Rates/Thresholds]`.
     - A FINCARDS-style structured reranker scores candidate cards by explicitly aligning the query's extracted `(Jurisdiction, Case Year, Tax Base)` against the card's structured fields, eliminating both **Wrong-Version (E3)** and **Wrong-Country (E2)** errors in a principled, generalizable manner.
  2. **Direction 2: Dual-Layer Thinking-Trace RAG (Agentic Solution Memory $\rightarrow$ Human Expert Test Set)**:
     - Adds a **second retrieval layer** over a corpus of verified step-by-step calculation traces (`gold_steps`).
     - **Layer 1 (Law Retrieval)**: Retrieves the applicable statutory articles from the official government corpus.
     - **Layer 2 (Thinking-Trace Retrieval)**: Retrieves 1–3 analogous worked step-by-step solutions from our **agent-generated dataset (2,400 verified questions with `gold_steps`)** to serve as structural reasoning exemplars when evaluating on **human-expert-created questions** (or cross-family questions).
- **Context Budgets**: Top-$k \in \{3, 5, 8, 12\}$ retrieved chunks.
- **Reasoning Modalities**: Both CoT (3.1) and Program-Aided CoT (3.2).

---

### Corpus Settings: Single-Country vs. Multi-Country Interference

To examine cross-border interference and retrieval robustness, Condition 3 is evaluated across two corpus scopes:

| Corpus Scope | Scope Description | Total Files | Key Analytical Objective |
| :--- | :--- | :---: | :--- |
| **A. Single-Country Corpus** | Contains only Kazakhstan's official laws (`data/kz/sources/`). | 53 files | Tests intra-jurisdictional ambiguity: distinguishing between the 2001, 2008, 2017, and 2025 Tax Codes and finding cross-referenced chapters. |
| **B. Multi-Country Corpus** | Aggregated corpus containing official laws from **all 7 countries** (`data/*/sources/`). | 205 files | Tests cross-border semantic interference: does the retriever erroneously pull Russian/Kazakh-translated VAT or CIT rules from other jurisdictions? |

---

## 5. Core Research Questions (RQs) and Hypotheses

The experimental evaluation tests four formal hypotheses:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE RESEARCH HYPOTHESES                        │
├────────────────────────────────────────────────────────────────────────┤
│ H1: Cross-Border Interference                                          │
│     In Multi-Country Corpus, Standard Semantic RAG suffers from        │
│     wrong-country retrieval, significantly lowering final calculation  │
│     accuracy compared to the Single-Country Corpus.                    │
├────────────────────────────────────────────────────────────────────────┤
│ H2: Perturbation Sensitivity & Family Brittleness                      │
│     LLM performance degrades across systematic variants:               │
│     Acc(Original) > Acc(Threshold) > Acc(Eligibility) > Acc(Temporal). │
│     Strict family consistency (all variants correct) will be <15%.     │
├────────────────────────────────────────────────────────────────────────┤
│ H3: Retrieval Architecture vs. Reasoning Synergy                       │
│     Hybrid + Reranking RAG substantially outperforms pure Dense RAG    │
│     in complete-evidence recall, while Program-Aided CoT eliminates    │
│     the pure arithmetic error ceiling seen in standard CoT.            │
├────────────────────────────────────────────────────────────────────────┤
│ H4: Epistemic Abstention vs. False Precision                           │
│     On missing-information cases, standard LLMs suffer from high false │
│     precision (hallucinating default values), whereas reasoning models │
│     achieve higher abstention accuracy (outputting "insufficient info")│
└────────────────────────────────────────────────────────────────────────┘
```

- **Primary Analytical Focus**: The percentage-point accuracy delta between **Condition 1 (Closed-Book)**, **Condition 2 (Oracle)**, and **Condition 3 (RAG)** under both CoT and Program-Aided reasoning.

---

## 6. Evaluation Measures & Scoring Protocols

The evaluation measures span numerical calculation, retrieval quality, intermediate reasoning, epistemic humility, and family consistency:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPREHENSIVE METRIC SUITE                      │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. Numerical Calculation       │ • Strict Exact Match (Acc_EM)         │
│                                │ • Lenient / Tolerant Match (Acc_tol)  │
│                                │ • Currency / Unit Match (Acc_unit)    │
├────────────────────────────────┼───────────────────────────────────────┤
│ 2. Retrieval Quality           │ • Recall@k of Gold Passages           │
│                                │ • Complete-Evidence Recall            │
│                                │ • Wrong-Country Retrieval Rate        │
│                                │ • Wrong-Version / Superseded Rate     │
│                                │ • Mean Reciprocal Rank (MRR)          │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. Reasoning Step Fidelity     │ • Gold Step Coverage (Step-Recall)    │
│                                │ • Intermediate Value Accuracy         │
│                                │ • Step-to-Answer Consistency          │
├────────────────────────────────┼───────────────────────────────────────┤
│ 4. Missing Information         │ • Correct Abstention Rate (Acc_abs)   │
│                                │ • False Abstention Rate (FAR)         │
│                                │ • Epistemic Macro-F1                  │
│                                │ • Missing Fact Semantic Accuracy      │
├────────────────────────────────┼───────────────────────────────────────┤
│ 5. Family & Temporal           │ • Strict Family Accuracy (Acc_fam-6)  │
│                                │ • Solvable Family Accuracy (Acc_fam-5)│
│                                │ • Temporal Decay Delta (Δ_temp)       │
│                                │ • Paired Temporal Consistency         │
└────────────────────────────────┴───────────────────────────────────────┘
```

### 6.1 Numerical Calculation Accuracy

For all 225 solvable evaluation questions ($y_{gold} \in \mathbb{R}$):

1. **Strict Exact Match ($Acc_{EM}$)**:
   $$\mathbb{I}_{EM} = \mathbb{I}\left( |\hat{y} - y_{gold}| \le \delta \right)$$
   where $\delta = 0$ for integer answers or statutory rounding, and $\delta = 0.01$ for floating currency values.
2. **Lenient / Tolerant Matching ($Acc_{tol}$)**:
   Accommodates minor intermediate floating-point rounding discrepancies:
   $$\mathbb{I}_{tol} = \mathbb{I}\left( \frac{|\hat{y} - y_{gold}|}{\max(|y_{gold}|, 1.0)} \le 0.01 \lor |\hat{y} - y_{gold}| \le 1.0 \right)$$
   (Accepts predictions within a $\pm 1\%$ relative margin or $\le 1.0$ currency unit).
3. **Unit Matching ($Acc_{unit}$)**:
   Verifies that the predicted unit $\hat{u}$ matches the gold unit $u_{gold}$ (`KZT`, `%`, etc.).

---

### 6.2 Retrieval Evaluation Metrics (for Condition 3 RAG)

1. **Recall@$k$**:
   $$\text{Recall@}k = \frac{|\text{Retrieved Passages}_{\le k} \cap \text{Gold Passages}|}{|\text{Gold Passages}|}$$
2. **Complete-Evidence Recall**:
   Measures whether **all** statutory provisions necessary to solve the calculation were retrieved:
   $$\mathbb{I}_{\text{complete}} = \mathbb{I}\left( \text{Gold Passages} \subseteq \text{Retrieved Passages}_{\le k} \right)$$
   *(Critical statutory metric: In multi-step tax calculations, retrieving 2 out of 3 required provisions typically leads to a completely wrong result).*
3. **Wrong-Country Retrieval Rate**:
   Percentage of retrieved top-$k$ passages originating from a jurisdiction other than Kazakhstan (measured in Multi-Country Corpus).
4. **Wrong-Version / Superseded Retrieval Rate**:
   Percentage of retrieved passages whose statutory applicability dates ($rule\_applies\_from, rule\_applies\_until$) do not overlap with the question's `{case_year}`.
5. **Mean Reciprocal Rank (MRR)**:
   Rank of the first applicable gold passage in the retrieved candidates.

---

### 6.3 Missing-Information & Epistemic Humility Metrics

Evaluated across the 45 missing-information variants and 225 regular questions:
1. **Correct Abstention Rate ($Acc_{abs}$)**:
   $$\text{Acc}_{abs} = \frac{1}{|Q_{missing}|} \sum_{q \in Q_{missing}} \mathbb{I}(\hat{y} = \text{"insufficient information"})$$
2. **False Abstention Rate ($FAR$ / Over-Conservatism)**:
   Percentage of complete, solvable questions where the model erroneously abstained.
3. **Epistemic Macro-F1**:
   Harmonic mean of precision and recall on the binary task of recognizing solvability vs. insufficiency.
4. **Missing Fact Explanation Score**:
   For questions where the model correctly abstained, evaluate whether its explanation identifies the omitted numeric fact using semantic similarity against ground-truth `missing_fact` and `final_answer_text`.

---

### 6.4 Family Consistency & Temporal Metrics

1. **Strict Family Consistency ($Acc_{fam-6}$)**: All 6 questions in the family solved correctly.
2. **Solvable Family Consistency ($Acc_{fam-5}$)**: All 5 regular calculation variants solved correctly.
3. **Temporal Decay Delta ($\Delta_{temp}$)**:
   $$\Delta_{temp} = Acc_{2020-2026} - Acc_{2000-2009}$$
4. **Paired Temporal Consistency**:
   Accuracy on matched pairs $(q_{modern}, q_{hist})$ derived from the same base scenario differing only by `{case_year}`.
5. **Statistical Significance via Family Block Bootstrapping**:
   Because variants within a family are statistically dependent, 95% confidence intervals and paired $p$-values are computed by **resampling complete families** rather than individual rows.

---

## 7. The 10-Tier Diagnostic Error Taxonomy

Every failed question is programmatically and qualitatively audited into a 10-tier failure taxonomy:

| Code | Failure Category | Diagnostic Criterion | Primary Attribution |
| :---: | :--- | :--- | :--- |
| **E1** | **Evidence Retrieval Failure** | Zero gold passages retrieved in top-$k$. | Retrieval system |
| **E2** | **Wrong-Country Retrieval** | Retrieved passages belong to a foreign jurisdiction. | Cross-border corpus noise |
| **E3** | **Incorrect Historical Version** | Retrieved passage is from a superseded or premature amendment. | Temporal blindness |
| **E4** | **Partial Evidence Retrieval** | Retrieved $\ge 1$ gold passage, but missed $\ge 1$ necessary provision. | Multi-provision retrieval gap |
| **E5** | **Classification Error** | Misclassified taxpayer status, residency, or business category. | Statutory interpretation |
| **E6** | **Threshold / Bracket Error** | Applied incorrect progressive bracket, cap, or marginal phase-out. | Statutory interpretation |
| **E7** | **Arithmetic / Rounding Error** | Correct statutory formula and rates, but incorrect arithmetic product. | Calculation failure |
| **E8** | **Unsupported Citation** | Model cited a fabricated or legally inapplicable article. | Hallucination |
| **E9** | **False Precision (Missing Info)** | Computed a number when essential information was missing. | Sycophancy / Lack of humility |
| **E10** | **Unnecessary Abstention** | Abstained on a complete, fully solvable regular question. | Over-conservatism |

---

## 8. Software Architecture & Execution Harness

The evaluation harness is designed as an isolated package inside `src/taxcalcbench/eval/`:

```
src/taxcalcbench/
├── eval/
│   ├── __init__.py
│   ├── config.py              # Configuration schemas (pilot KZ, dev/test split, conditions, k)
│   ├── dataset.py             # Loads outputs/full/kz/questions.json; partitions 5 dev / 45 test
│   ├── runner.py              # Orchestrates evaluation runs across conditions via Gemini API
│   ├── prompts.py             # Standardized prompts for CoT and Program-Aided CoT
│   ├── extractors.py          # Robust regex extractors for \boxed{...}, units, and abstentions
│   ├── sandbox.py             # Safe, isolated Python execution runtime for Program-Aided CoT
│   ├── metrics.py             # Computes Acc_EM, Acc_tol, Recall@k, Complete-Evidence, FAR, bootstrap CI
│   ├── audit.py               # Automated 10-tier error classifier
│   └── rag/
│       ├── chunker.py         # Statutory section/article-aware chunker for data/kz/sources
│       ├── indexer.py         # Builds BM25 (Snowball ru) and BGE-M3 vector stores
│       └── retriever.py       # Implements Lexical, Dense, Hybrid (RRF), and Reranked retrieval
```

### 8.1 Safe Execution Sandbox (Program-Aided CoT)
For Program-Aided CoT conditions (1.2, 2.2, 3.2):
- Generated Python code is executed in an isolated runtime:
  - 5-second hard execution timeout.
  - Memory capped at 512MB.
  - Prohibited modules: `os`, `sys`, `subprocess`, `socket`, `shutil`, `urllib`.
  - Pre-imported exact libraries: `decimal.Decimal`, `math`, `datetime`.
  - Captures the standard output and terminal `ans` variable.

### 8.2 Standardized Output Contract
Prompts enforce deterministic formatting for automated grading:

```text
[CONTEXT]:
{retrieved_or_gold_passages}

[QUESTION]:
{question_text}

[INSTRUCTIONS]:
Solve this tax calculation for Kazakhstan in the year {case_year}.
Think step by step, citing applicable legal provisions and performing exact calculations.
Conclude your response with the final answer on the last line using this exact format:
[FINAL ANSWER]: \boxed{value} [UNIT]
(If essential numeric facts are missing and the calculation cannot be completed, write:
[FINAL ANSWER]: \boxed{insufficient information})
```

---

## 9. Phased Execution Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHASED WORK PLAN: FROM PILOT TO FULL BENCHMARK               │
└─────────────────────────────────────────────────────────────────────────────────┘

  Stage 1: Pilot Preparation & Development Tuning (Kazakhstan Dev Set)
  ├── 1.1 Implement dataset partitioner (5 dev families / 45 test families in KZ)
  ├── 1.2 Implement statutory chunker on data/kz/sources/ (preserving Article boundaries)
  ├── 1.3 Build retrieval index (BM25 with Snowball Russian stemmer + BGE-M3 embeddings)
  ├── 1.4 Test on 5 Dev Families (30 questions) with Gemini 3.8 Flash
  └── 1.5 Calibrate regex extraction, Python sandbox timeouts, and k in {3, 5, 8}

  Stage 2: Single-Country Pilot Execution (Kazakhstan 45 Test Families)
  ├── 2.1 Run Condition 1 (Closed-Book CoT & Program-Aided) on 270 test questions
  ├── 2.2 Run Condition 2 (Oracle Grounding CoT & Program-Aided) on 270 test questions
  ├── 2.3 Run Condition 3 (RAG CoT & Program-Aided) across BM25, Dense, Hybrid, Reranked
  ├── 2.4 Test Single-Country vs. Multi-Country corpus interference
  ├── 2.5 Run comparative benchmark with Gemini 2.5 Pro for model capacity comparison
  └── 2.6 Execute automated grading (Acc_EM, Acc_tol, Recall@k, Complete-Evidence, FAR)

  Stage 3: Error Audit & Setup Freezing
  ├── 3.1 Classify all failure cases using the 10-Tier Error Taxonomy
  ├── 3.2 Verify statistical stability using family-block bootstrap resampling
  └── 3.3 Freeze prompt templates, retrieval depth, chunking, and scoring parameters

  Stage 4: Multi-Country Benchmark Rollout
  ├── 4.1 Index official sources for remaining 6 countries (CN, PL, ID, PK, EG, IN)
  ├── 4.2 Execute evaluation pipeline on remaining 1,620 test questions
  └── 4.3 Aggregate cross-jurisdictional and cross-temporal comparison tables

  Stage 5: Scientific Reporting & Publication
  ├── 5.1 Compile comparative results, radar charts, and temporal decay graphs
  └── 5.2 Draft manuscript and release open benchmark suite
```

---

## 10. Summary of Key Methodological Strengths

1. **Grounded Empirical Reality**: Evaluates models on 2,100 ground-truth questions and 14,307 primary citations rather than synthetic or commercial web text.
2. **Pragmatic and Resource-Efficient**: Standardizing on the Gemini API with a single-country pilot (Kazakhstan) eliminates infrastructure bottlenecks and enables rapid, low-cost iteration.
3. **Ablation of Arithmetic vs. Legal Bottlenecks**: Combining CoT with Program-Aided Python execution isolates calculation errors from statutory interpretation errors.
4. **Isolating Retrieval from Reasoning**: The Oracle condition establishes the exact ceiling for legal reasoning, while Complete-Evidence Recall pinpoints multi-hop retrieval bottlenecks.
5. **Epistemic Humility Auditing**: The missing-information evaluation measures whether LLMs can safely identify ambiguity and prevent real-world tax calculation hallucinations.

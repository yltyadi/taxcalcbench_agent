# TaxCalcBench Agent — Comprehensive Project Context Window

*This document serves as the complete, authoritative context window and specification for any coding agent working on, extending, or evaluating the TaxCalcBench automated tax benchmark generation and experimental evaluation system.*

---

## 1. Project Objective & Core Mandate

The primary objective of **TaxCalcBench** is twofold:
1. **Benchmark Generation**: Build, execute, debug, and monitor an automated multi-agent system that produces auditable, legally sound tax calculation benchmark datasets across multiple countries based on official national laws, the 6-category OECD classification, and multi-period coverage.
2. **Experimental Evaluation (Current Phase)**: Execute controlled empirical evaluations of frontier LLMs (specifically Google Gemini via API) across multiple retrieval conditions (Closed-book, Oracle grounding, Multi-retriever RAG) and reasoning modalities (Chain-of-Thought vs. Program-Aided Python execution) based on **`EXPERIMENT_DESIGN.md`**.

### Benchmark Scope & Datasets:
- **7 Countries / 8 Language Datasets**:
  1. **Kazakhstan (`KZ` - Russian)**: Russian (`ru`) — 50 families, 300 questions, 2,578 citations.
  2. **Kazakhstan (`KZ` - Kazakh)**: Kazakh (`kk`) — 50 families, 300 questions, 2,578 citations.
  3. **China (`CN`)**: Chinese (`zh`) — 50 families, 300 questions, 2,807 citations.
  4. **Poland (`PL`)**: Polish (`pl`) — 50 families, 300 questions, 2,067 citations.
  5. **Indonesia (`ID`)**: Indonesian (`id`) — 50 families, 300 questions, 2,179 citations.
  6. **Pakistan (`PK`)**: English (`en`) — 50 families, 300 questions, 1,549 citations.
  7. **Egypt (`EG`)**: Arabic (`ar`) — 50 families, 300 questions, 1,326 citations.
  8. **India (`IN`)**: English (`en`) — 50 families, 300 questions, 1,801 citations.
- **Dataset Size per Country/Language**: Exactly **50 families** (300 questions).
- **Questions per Family**:
  - **5 regular numeric calculation questions** across distinct variants (original, temporal, threshold, eligibility, numbers).
  - **1 derived missing-information question** (where one essential numeric fact is omitted; answer is strictly `"insufficient information"`).
  - **Total**: 6 questions per family $\times$ 50 families = **300 questions per country** (Grand Total: **2,400 questions** and **16,885 auditable citations** across all 8 datasets).

---

## 2. Benchmark Rules, Taxonomy & Quality Standards

### A. Strict Evidence Standard (Official National Law Only)
- **Primary Source Requirement**: Only official government legislation (national Acts, Presidential Decrees, Government Regulations, Official Gazettes, or Revenue Authority publications). Commercial sites, blogs, and unverified summaries are strictly prohibited.
- **Verbatim Citations**: Every single step in the reasoning chain must reference an exact, contiguous passage (`supporting_passage`) quoted directly from the official text in its native language, matching `schema._source_quote`.
- **Rule Boundary Dates**: `rule_applies_from` and `rule_applies_until` in `YYYY-MM-DD` format where established by the text.

### B. Multi-Period Temporal Quotas
Every country's 250 regular questions satisfy a mandatory temporal distribution across three historical eras:
1. **Contemporary (2020–2026)**: **60%** (150 regular questions + 50 missing-info = 200 questions).
2. **Intermediate (2010–2019)**: **20%** (50 regular questions).
3. **Historical (2000–2009)**: **20%** (50 regular questions).
- **Family Multi-Period Rule**: Every family covers at least 2 distinct periods (`min_periods_per_family: 2`).
- **Temporal Counterparts**: Every family contains at least one temporal pair: a regular question derived from an earlier question in another period, sharing identical wording and scenario facts except for `{case_year}`, and legally feasible in both years.

### C. 6-Category OECD Tax Classification
Every family maps to an authoritative OECD tax category:
- `1000`: **Taxes on income, profits and capital gains** (PIT, CIT, capital gains, withholding).
- `2000`: **Social security contributions** (Compulsory pension, health insurance, social funds).
- `3000`: **Payroll and workforce taxes** (Unrequited employer wage-bill levies).
- `4000`: **Property taxes** (Immovable property, land transfer duties, wealth taxes).
- `5000`: **Goods and services taxes** (VAT, GST, turnover taxes, excises).
- `6000`: **Other taxes** (Stamp duties, statutory rounding, vehicle/mineral levies).

### D. Missing-Information Question Contract
- Derived strictly from an approved parent case in the same family.
- Exactly one material numeric input is omitted from the question text (replacing it with natural phrasing).
- Short answer (`answer_value`) must be strictly `"insufficient information"`.
- Unit must be `null`.
- `final_answer_text` must name the missing fact and explain why it prevents a unique calculation.
- Inherits the verified citations of its parent case.

---

## 3. Benchmark Status & Validated Datasets Inventory

As of October 2026, **all datasets are 100% complete and validated with 0 errors**:

| Dataset / Country | Language | Status | Validated Families | Validated Questions | Auditable Citations | Official Corpus |
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

- **Unit & Regression Tests**: **561 passed in 15.80s** (`pytest`).

---

## 4. Codebase Structure & Directory Layout

```text
taxcalcbench_agent/
├── EXPERIMENT_DESIGN.md       # Comprehensive experimental evaluation framework & research plan
├── EXPERIMENT_STATUS.md       # Live evaluation progress, metrics, and bilingual KZ benchmark status
├── CONTEXT_WINDOW.md          # Authoritative system specification and context window
├── configs/                   # Country configuration files
│   ├── cn.json                # China configuration
│   ├── eg.json                # Egypt configuration
│   ├── id.json                # Indonesia configuration
│   ├── in.json                # India configuration
│   ├── kz.json                # Kazakhstan (Russian) configuration
│   ├── kz_kk.json             # Kazakhstan (Kazakh) configuration
│   ├── pk.json                # Pakistan configuration
│   └── pl.json                # Poland configuration
├── data/                      # Official downloaded government sources
│   ├── cn/sources/            # China official statutes & texts
│   ├── eg/sources/            # Egypt transcribed & extracted Arabic laws
│   ├── id/sources/            # Indonesia laws & texts
│   ├── in/sources/            # India 1961 Act, 1962 Rules, GST Acts
│   ├── kz/sources/            # Kazakhstan Russian official Tax Codes & decrees
│   ├── kz_kk/sources/         # Kazakhstan Kazakh official Tax Codes & decrees (from Adilet)
│   ├── pk/sources/            # Pakistan ITO 2001, Sales Tax Act, etc.
│   └── pl/sources/            # Poland ELI HTML laws
├── outputs/full/              # Final exported datasets & checkpoints
│   ├── cn/                    # China questions.json, citations.json, work/
│   ├── eg/                    # Egypt questions.json, citations.json, work/
│   ├── id/                    # Indonesia questions.json, citations.json, work/
│   ├── in/                    # India questions.json, citations.json, work/
│   ├── kz/                    # Kazakhstan Russian questions.json, citations.json, work/
│   ├── kz_kk/                 # Kazakhstan Kazakh questions.json, citations.json, work/
│   ├── pk/                    # Pakistan questions.json, citations.json, work/
│   └── pl/                    # Poland questions.json, citations.json, work/
├── prompts/                   # System instructions for benchmark generation agents
│   ├── plan.md                # Planning instructions
│   ├── create.md              # Question writing instructions
│   ├── review.md              # Independent review instructions
│   └── derive_missing.md      # Missing-information derivation instructions
├── scripts/                   # Specialized extraction & conversion utilities
│   ├── extract_egypt.py       # Multimodal OCR pipeline for Egyptian Arabic PDFs
│   ├── download_kazakh_sources.py  # Downloader & HTML parser for official Kazakh Adilet laws
│   ├── translate_kazakh_pipeline.py # Rerunnable agentic RU -> KK translation & citation alignment
│   └── run_eval.py            # CLI evaluator for Closed-Book and Oracle Grounding experiments
├── src/taxcalcbench/          # Core implementation package
│   ├── eval/                  # Experimental evaluation harness (runner, dataset, extractors, metrics)
│   ├── cli.py                 # Command-line entry points
│   ├── experts.py             # Agent definitions, tools, metering, and retries
│   ├── pipeline.py            # Main generation, review, and export orchestrator
│   ├── schema.py              # Pydantic schemas, validation, quote matcher
│   ├── settings.py            # Assignment and category distribution logic
│   ├── sources.py             # Ingestion, PDF/HTML extractors, policy checks
│   └── tools.py               # Calculation and statutory search tools
├── tests/                     # 561 unit and regression tests
└── docs/                      # Architectural documentation and source notes
```

---

## 5. Experimental Evaluation Architecture (`EXPERIMENT_DESIGN.md`)

When executing experiments on the HPC cluster, any agent must follow the methodology set out in **`EXPERIMENT_DESIGN.md`**.

### A. The 3 Retrieval Regimes $\times$ 2 Reasoning Modalities Matrix

Every evaluation question is tested across a structured $3 \times 2$ grid:
1. **Condition 1: Closed-Book (No Retrieval / Parametric Memory)**
   - *1.1 Closed-Book CoT*: Step-by-step reasoning from parametric memory alone.
   - *1.2 Closed-Book Program-Aided CoT*: Generates and executes exact Python arithmetic in an isolated sandbox.
2. **Condition 2: Oracle Grounding (Gold Evidence Upper Bound)**
   - *2.1 Oracle CoT*: Injected with verbatim gold statutory passages from `citations.json`.
   - *2.2 Oracle Program-Aided CoT*: Receives gold statutory provisions and executes Python code.
   - *Objective*: Isolates legal comprehension and math from retrieval failure.
3. **Condition 3: Multi-Retriever RAG (Official Statute Corpus)**
   - Benchmarks four retrieval architectures:
     - **Lexical (BM25)**: Tuned with language stemmers (Snowball for Russian/Kazakh).
     - **Dense Semantic**: Multilingual embeddings (`BGE-M3` or Google `text-embedding-004`).
     - **Hybrid**: Reciprocal Rank Fusion (RRF) combining BM25 + Dense.
     - **Hybrid + Cross-Encoder Reranker**: Top candidates reranked using `bge-reranker-large`.
   - Context budgets: Top-$k \in \{3, 5, 8\}$ retrieved chunks.
   - Tested under both **CoT (3.1)** and **Program-Aided CoT (3.2)**.

### B. Corpus Context Scopes
- **Single-Country Corpus**: Restricts retrieval to the target country's sources (e.g., `data/kz/sources/` or `data/kz_kk/sources/`). Isolates intra-country temporal version conflicts.
- **Multi-Country Corpus**: Aggregates all 205 legislative files across all 7 countries (`data/*/sources/`). Tests cross-border interference and wrong-country retrieval.

### C. Development vs. Evaluation Split (Family-Preserving)
For the pilot country (Kazakhstan `KZ`):
- **Development Set (Dev)**: Exactly **5 Families (30 questions)** (`A-F01` to `A-F05`).
  - Covers Corporate Income Tax, Immovable Property Tax, Value Added Tax, Social Tax, and Personal Income Tax across all 3 periods.
  - Used strictly for tuning chunk sizes, prompt formats, regex extractors, and Python sandbox timeouts.
- **Evaluation Set (Test)**: Exactly **45 Families (270 questions)** (`A-F06` to `A-F50`).
  - Held strictly frozen for the benchmark report.
- **No Variant Splitting**: All 6 variants of each family stay clustered together to eliminate data contamination.

### D. Model Standardization
- Standardized on **Google Gemini models via API calls**:
  - `gemini-3.8-flash`: Fast, efficient primary baseline (Temperature 0.0, max tokens 8,192).
  - `gemini-2.5-pro`: Deep reasoning comparison model (Temperature 0.0, max tokens 16,384, reasoning effort enabled).
- Checkpoints are saved per question to disk (`outputs/eval/<country>/work/<condition>/<question_no>.json`) to guarantee 100% resume capability with zero duplicate token spend.

### E. Evaluation Metrics Suite
1. **Numerical Calculation**:
   - Strict Exact Match ($Acc_{EM}$): Exact value match under statutory rounding rules.
   - Lenient / Tolerant Match ($Acc_{tol}$): Relative margin $\le 1\%$ or absolute margin $\le 1.0$.
   - Unit Match ($Acc_{unit}$): Verifies currency unit (`KZT`, `%`, etc.).
2. **Retrieval Quality**:
   - Recall@$k$ of gold passages.
   - **Complete-Evidence Recall**: Binary check whether *all* statutory provisions required for the multi-step calculation were retrieved.
   - Wrong-Country Retrieval Rate & Wrong-Version / Superseded Rate.
   - Mean Reciprocal Rank (MRR).
3. **Reasoning Step Fidelity**:
   - Step-by-step alignment with ordered `gold_steps` (intermediate values and citation support).
4. **Missing Information (Epistemic Humility)**:
   - Correct Abstention Rate ($Acc_{abs}$) on missing-info variants (answer = `"insufficient information"`).
   - False Abstention Rate ($FAR$) on solvable cases (over-conservatism).
   - Epistemic Macro-F1.
5. **Family Consistency & Statistical Testing**:
   - Strict Family Accuracy ($Acc_{fam-6}$) and Solvable Family Accuracy ($Acc_{fam-5}$).
   - 95% Confidence Intervals via **block bootstrap resampling across complete families**.
6. **10-Tier Diagnostic Error Taxonomy**:
   - E1 (Evidence Retrieval Failure), E2 (Wrong-Country Retrieval), E3 (Incorrect Historical Version), E4 (Partial Evidence Retrieval), E5 (Classification Error), E6 (Threshold/Bracket Error), E7 (Arithmetic/Rounding Error), E8 (Unsupported Citation), E9 (False Precision on Missing Info), E10 (Unnecessary Abstention).

### F. Completed Evaluation Results (Kazakhstan Bilingual Pilot & Step 2)
- **Dev Set (30 questions)**: Closed-Book Exact Match 68.0% (RU) / 64.0% (KK) $\rightarrow$ Oracle Grounding 100.0% (RU) / 96.0% (KK). Program-Aided CoT: Closed-Book 60.0% $\rightarrow$ Oracle Grounding 96.0%.
- **Test Set (270 questions)**: Closed-Book Solvable Exact Match 47.6% (RU) / 49.3% (KK) $\rightarrow$ Oracle Grounding 93.8% (RU) / 92.0% (KK) ($\Delta = +46.2\%$ over closed-book). Family Consistency: 13.3% $\rightarrow$ 77.8%.
- **Bilingual Parity**: Performance between Russian and Kazakh matches within $\pm 2\%$, confirming statutory tax calculations are solvable in native Kazakh when grounded in Adilet statutes.
- **Recency Decay**: Closed-book accuracy collapses from 55.6% (2020–2026) down to 26.7% (2000–2009) ($\Delta_{\text{temp}} = 28.9\%$). Oracle grounding restores 2000s accuracy to 86.7%.
- Full live audit logs, metric tables, and bilingual KZ methodologies are maintained in **`EXPERIMENT_STATUS.md`**.

---

## 6. HPC Cluster Environment & Execution Protocol

### A. Environment Specifications
- **Login Node**: `lo-01` (`login-student-lab.mbzu.ae`)
- **Compute Nodes**: `ws-ia` (interactive partition), `ws-l1-*`, `ws-l5-*`
- **Project Directory**: `/home/adi.yeltay/projects/taxcalcbench_agent`
- **Conda Environment**: `~/.conda/envs/taxcalcbench/bin/python`
  - Activate via:
    ```bash
    source /apps/local/anaconda3/conda_init.sh
    conda activate taxcalcbench
    ```
- **API Credentials**: Placed in `.env`:
  `GOOGLE_API_KEY=...` or `GEMINI_API_KEY=...`

### B. Slurm Session & Walltime Management
Interactive Slurm allocations (`salloc` / `srun`) on partition `ws-ia` hit a 24-hour walltime limit:
```bash
# 1. Start tmux on the LOGIN node FIRST:
tmux new -s taxcalc_eval

# 2. Inside tmux, request the compute allocation:
salloc -N1 --time=24:00:00 --mem=32GB

# 3. Inside the allocated compute node, activate Conda:
source /apps/local/anaconda3/conda_init.sh
conda activate taxcalcbench
cd /home/adi.yeltay/projects/taxcalcbench_agent

# 4. Run commands inside this protected tmux session.
# If disconnected, reconnect using:
tmux attach -t taxcalc_eval
```

### C. Data Transfer Protocol (Mac <-> HPC)

#### 1. Transferring Kazakh Dataset (`KZ_kk`) to HPC:
The archive `kz_kk_bundle.tar.gz` (7.3 MB) contains `outputs/full/kz_kk` and `data/kz_kk`:
- **On Mac**:
  ```bash
  scp kz_kk_bundle.tar.gz adi.yeltay@login-student-lab.mbzu.ae:/home/adi.yeltay/projects/taxcalcbench_agent/
  ```
- **On HPC**:
  ```bash
  cd /home/adi.yeltay/projects/taxcalcbench_agent
  tar -xzvf kz_kk_bundle.tar.gz
  ```

#### 2. Syncing Git Changes:
Code files, configs, and documentation (`EXPERIMENT_DESIGN.md`, `configs/kz_kk.json`, `scripts/`) are tracked in git:
- **On Mac**:
  ```bash
  git add EXPERIMENT_DESIGN.md configs/kz_kk.json scripts/
  git commit -m "Add experiment design and Kazakh pipeline scripts"
  git push origin main
  ```
- **On HPC**:
  ```bash
  git pull origin main
  ```

#### 3. Backing Up Evaluation Outputs from HPC to Mac:
```bash
# On HPC:
tar -czvf eval_results.tar.gz outputs/eval/

# On Mac:
scp adi.yeltay@login-student-lab.mbzu.ae:/home/adi.yeltay/projects/taxcalcbench_agent/eval_results.tar.gz .
tar -xzvf eval_results.tar.gz
```

---

## 7. Technical Decisions & Chronic Issues Resolved

### A. Provider Error Handling & Tool Call Collisions
- **HTTP 500 / 503 / 429 & Connection Resiliency**: Configured `max_retries: 15` and native exponential backoff up to 120s in `_retry_provider` within `src/taxcalcbench/experts.py`.
- **Gemini Tool Call ID Collisions**: Gemini occasionally emits duplicate `call_id` strings across consecutive chunks in the OpenAI compatibility layer, causing the Agents SDK to throw `ModelBehaviorError`. Fixed by deduplicating tool call IDs in `_MeteredModel` by tracking `seen_call_ids` and appending unique suffixes.

### B. Citation Quotation Normalization (`schema.py`)
- Cleaned null and control byte artifacts before running whitespace normalization:
  ```python
  clean_quote = quote.replace("\x00\x00a0", " ").replace("\x00", " ")
  clean_quote = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]+", " ", clean_quote)
  target = re.sub(r"\s+", " ", clean_quote).strip()
  ```

### C. Regex Parenthesis Handling in Statutory Aligners
- When statutory article identifiers contain complex multi-parenthetical annotations (e.g. `Статья 1, пункт 14 (статья 25)`), unescaped string interpolation causes `re.PatternError: unbalanced parenthesis`.
- **Resolution**: Implemented core article extraction (`re.findall(r"[0-9]+(?:-[0-9]+)?", art_sec)`) and escaped patterns using `re.escape()` before compiling regexes in `scripts/translate_kazakh_pipeline.py`.

### D. Citation Cache Integrity Guardrails
- Evicted any malformed empty-quote cache entries in `citation_cache.json`.
- Added strict non-empty string assertions in `align_unique_citation` before committing to cache, ensuring every citation satisfies `CitationRow` minimum length requirements and passes `_source_quote`.

---

## 8. CLI & Execution Commands Reference

```bash
# 1. Health checks & Settings inspection
python -m taxcalcbench doctor --country configs/kz.json
python -m taxcalcbench settings --country configs/kz.json
python -m taxcalcbench doctor --country configs/kz_kk.json

# 2. Run benchmark generation for a single country (if needed)
python -m taxcalcbench run \
  --country configs/kz.json \
  --source-dir data/kz/sources \
  --output outputs/full/kz

# 3. Run Kazakh translation & citation alignment pipeline (resumable)
python scripts/translate_kazakh_pipeline.py
python scripts/translate_kazakh_pipeline.py --family A-F01
python scripts/translate_kazakh_pipeline.py --export-only

# 4. Run experimental evaluations across any country & condition
# Full dataset evaluation (all 300 questions, default):
python scripts/run_eval.py --country in --split full --conditions closed_book_cot oracle_cot
python scripts/run_eval.py --country eg --split full --conditions closed_book_cot oracle_cot
python scripts/run_eval.py --country pl --split full --conditions closed_book_cot oracle_cot
python scripts/run_eval.py --country kz --split full --conditions closed_book_cot oracle_cot

# Program-Aided CoT (Python execution sandbox):
python scripts/run_eval.py --country kz --split dev --conditions closed_book_program_cot oracle_program_cot

# Category-balanced dev or test subsets:
python scripts/run_eval.py --country kz --split dev --conditions closed_book_cot oracle_cot
python scripts/run_eval.py --country kz_kk --split test --conditions closed_book_cot oracle_cot

# Custom families or quick test limits:
python scripts/run_eval.py --country in --split full --limit 10
python scripts/run_eval.py --country cn --families A-F01 A-F02 A-F10

# 5. Verify questions and citations integrity locally
python -c "
import json
for c in ['cn', 'eg', 'id', 'in', 'kz', 'kz_kk', 'pk', 'pl']:
    q = json.load(open(f'outputs/full/{c}/questions.json'))
    cit = json.load(open(f'outputs/full/{c}/citations.json'))
    print(f'{c}: {len(q)} questions, {len(cit)} citations')
"
```

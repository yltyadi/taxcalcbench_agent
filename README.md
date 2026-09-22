# Tax question creator

Create tax-calculation question families from official national laws, using **OpenAI Agents SDK + Gemini** and local files.

The workflow is **download sources → plan one family → write questions → one independent review → optional missing-information questions → export**. Each family finishes before the next starts. There is no database or experiment framework.

## Install

Use Python 3.11–3.13:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
```

Put `GOOGLE_API_KEY=your_key` in the ignored `.env` file. `GEMINI_API_KEY` also works. No OpenAI key is needed. The configured model is `gemini-3.8-flash`.

## Settings

Edit the relevant file in [configs/](configs/): `kz.json` (Russian), `pk.json` and `in.json` (English), `cn.json` (Chinese), `eg.json` (Arabic), `id.json` (Indonesian), or `pl.json` (Polish). Each contains its own explicit official source list. The Python code contains no country tax formulas. Source availability and extraction checks are documented in [docs/](docs/); a configuration alone does not guarantee every historical dependency is available.

| Setting | Default |
| --- | --- |
| `language` | `ru` for Kazakhstan |
| `generation.families` | 50 |
| `generation.regular_questions_per_family` | 5 |
| `generation.missing_information_per_family` | 1; use 0 to disable |
| `generation.min_periods_per_family` | 2 |
| `generation.periods` | 2020–2026, 2010–2019, 2000–2009; weights 3:1:1 |
| `generation.categories` | Six OECD categories with equal starting weights |
| `runtime` | Model, response size and timeout; writer/reviewer use high reasoning in the supplied config |
| `sources` | Approved hosts/paths, official URLs and download settings |

The defaults produce **250 regular + 50 missing-information questions**. Regular year-period targets are 150/50/50; 25 families give 75/25/25. Extra questions do not affect those ratios. Every family includes a temporal counterpart with the same scenario except its year, using the applicable law independently. Category weights are starting targets; the planner may make a justified reallocation using available law while preserving temporal balance.

Preview settings without downloads or model calls:

```sh
.venv/bin/taxcalcbench settings --country configs/kz.json
```

## Run

One family, including its optional extra:

```sh
.venv/bin/taxcalcbench run \
  --country configs/kz.json \
  --single-family \
  --source-dir data/kz/temporal-sources \
  --output outputs/kz-smoke
```

Full configured dataset, or resume the existing full run:

```sh
.venv/bin/taxcalcbench run \
  --country configs/kz.json \
  --source-dir data/kz/temporal-sources \
  --output outputs/kz-full
```

Use `--families N`, `--regular-per-family N` or `--missing-per-family N` for a different target. A changed dataset target needs a separate output folder. Otherwise, **repeat the same command to resume**; saved approvals and unfinished feedback are retained.

Corrections are automatic. A reviewer rejection returns concrete feedback to the writer, or to the planner if the scenario needs revision. A temporal wording conflict returns both questions to the writer before review, preserving their drafts and the rest of the family. Shared scenarios must work in both years; changing statutory dates or rates belong in each year's solution when they are not needed as factual inputs. A missing variant change note uses the planner's existing explanation instead of triggering a rewrite. A repeated unchanged rejected draft remains pending instead of spinning. No correction-count setting is needed.

Temporary connection, rate-limit and server errors retry the same request through the SDK, retaining the current tool history. If retries are exhausted, the run saves progress and pauses on that family instead of failing subsequent families. A paused/partial run exits with code 2. There is no default cumulative token, model-call or agent-turn cutoff; usage is recorded in `work/usage.json`.

Connection recovery uses eight retries by default, with exponential waits starting at five seconds and capped at sixty seconds; provider `Retry-After` instructions can be longer. Set `runtime.max_retries` in the country JSON to change this. These are retries for one failed request, not a limit on successful generation. A wrong assigned case/year receives one focused regeneration with explicit feedback; its answer is never silently relabeled.

If a writer returns the actual year instead of `{case_year}`, an exact match to the known template is recovered locally without another model call. Other missing-placeholder cases enter the normal correction loop. Year digits in amounts or historical facts are never replaced automatically.

## Several countries

Run the four additional countries that passed the live family check sequentially, each with its own sources, output and resume state:

```sh
python -m taxcalcbench run \
  --countries configs/pk.json configs/cn.json configs/id.json configs/pl.json \
  --source-root data \
  --output-root outputs/full
```

Sources go to `data/<code>/sources/`; results go to `outputs/full/<code>/`. Kazakhstan is not included. Rerun the same command to resume: completed families make no new model calls. One country's partial/error result does not prevent later countries from starting, and the command exits with code 2 if any country is incomplete. Ctrl+C stops the sequence. For a small end-to-end check, add `--single-family` and use a separate `--output-root outputs/smoke`. The original `--country ... --source-dir ... --output ...` command remains supported.

Egypt's Arabic configuration is included but generation is explicitly blocked by `generation.blocked_reason`: its readable official documents do not cover the required historical periods. Downloads, settings and diagnostics remain available. Including `configs/eg.json` in a batch reports `blocked` without model calls and continues to the other countries; the batch then exits with code 2. Resolve the documented source problem before removing the block.

India's English configuration and nine readable official documents are included, but its live family test stopped on a missing 2000s income-tax rate schedule. Its `generation.blocked_reason` also prevents paid generation until that source gap is resolved; see the [India source findings](docs/sources-pk-in.md). Download and diagnostic commands remain available. Remove that setting only after preparing the missing evidence. The required temporal coverage has not been reduced.

## MBZUAI HPC / Ubuntu

The supplied lab guide uses **Slurm** (`salloc`/`sbatch`). Allocate a workstation before running the program; do not run generation on the login node. This is an API workload: no GPU is required, but the allocated workstation must reach Gemini over HTTPS. Source preparation also needs access to the configured official websites.

Use persistent shared storage for this checkout, `data/` and `outputs/`. Initialize the lab's conda installation, then create a Linux environment (Ubuntu's system Python 3.10 is too old):

```sh
source /apps/local/anaconda3/conda_init.sh
conda create -n taxcalcbench -c conda-forge python=3.12 poppler -y
conda activate taxcalcbench
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

Use the alternative conda initialization path from your lab guide if needed. Poppler supplies `pdftotext`, required by the India and Egypt PDF configurations; it needs no sudo when installed through conda. Set the Gemini key in the ignored `.env` file or the environment, never in the job script. Do not copy the macOS `.venv` to Linux.

Interactive allocation:

```sh
tmux new -s taxcalcbench
salloc --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
srun --pty bash
# On the allocated workstation: activate your environment and enter the project.
python -m taxcalcbench doctor --country configs/pk.json --live
python -m taxcalcbench run --country configs/pk.json --output outputs/full/pk
# Or use the multi-country command above.
```

Alternatively, from the project directory with the environment activated:

```sh
sbatch scripts/hpc.slurm configs/pk.json configs/cn.json configs/id.json configs/pl.json
```

[scripts/hpc.slurm](scripts/hpc.slurm) requests two CPU cores, 8 GB RAM and twelve hours; adjust the time, partition/account and resource request to the lab's allocation policy. It runs the countries sequentially and writes `taxcalcbench-<jobid>.log`. Set `TAXCALCBENCH_OUTPUT_ROOT=outputs/smoke` and append `--single-family` for a small job. `TAXCALCBENCH_DATA_ROOT` and `TAXCALCBENCH_PYTHON` can override the storage root and Python executable. Check a model connection from the allocated workstation, since a successful login-node connection does not establish compute-node access.

Slurm termination preserves previously completed checkpoints. Resubmit the same command after a wall-time limit or temporary Gemini outage. Do not run two jobs against the same output directory. No retry can fix a permanently blocked network route or invalid credentials.

Interrupted source preparation also resumes: retained original/text checksums are verified first, and only unfinished URLs are fetched. Source files publish atomically so an interrupted write is not mistaken for a complete document. For connection failures, `work/operations/*/provider_errors/` records redacted transport details (for example DNS or TLS errors); check these before repeatedly resubmitting a job that has no outbound network access.

A completed download batch also records unavailable URLs in `sources.json`; rerunning generation does not retry those failed downloads. Inspect their recorded errors and use `download --refresh` to prepare a new source collection after access is restored. Use that collection with a new output directory if its evidence differs from an existing run.

## Moving or preserving runs

Both `data/` and `outputs/` are Git-ignored runtime artifacts; ignoring them does not delete their local contents. A fresh country run creates/downloads its source collection automatically. To continue an existing run, transfer its **entire source collection and entire output directory, including `work/`**, separately from Git after the process has stopped writing. The exact evidence files and checksums are required; exports alone cannot resume work.

Paths can change between machines: resume now compares source identities and checksums, not absolute directory names. No manual checkpoint edits are required. Keep the same dataset configuration, reference files and original source bytes. Completed KZ artifacts can remain archived locally and do not need to be copied when starting the six new countries.

## Official sources

`run` prepares sources before generation. Download them separately if desired:

```sh
.venv/bin/taxcalcbench download \
  --country configs/kz.json \
  --source-dir data/kz/temporal-sources
```

Downloads follow configured official national URLs and finite link rules in stable order. All redirects must satisfy the allowlist. Original files, extracted text and checksums are retained. The Kazakhstan list includes code editions, commencement/amendment laws, budgets and supporting legislation spanning the configured periods.

**The corpus stays fixed during question creation.** Agents can read and search these local laws but cannot browse or download additional sources. If a dependency is unavailable, the planner chooses a supported scenario or year within the assigned period. A remaining gap is reported; no values are invented to fill it.

HTML, readable DOCX and text PDFs are supported. Unreadable provisions remain excluded. A file's adoption or snapshot date is not by itself proof that its rules apply to a question year.

`--max-documents 2` is only a download smoke-test override and is inadequate for temporal question generation. Normal commands omit it. Explicit additions to the official-source configuration can be downloaded before a resumed run while preserving existing source bytes. Replacing evidence requires a new source collection (`download --refresh`).

## Results and checks

```text
outputs/<run>/
  questions.json    approved regular and ready derived questions
  citations.json    their official-law citations
  work/             plans, drafts, review results, progress and usage
```

The two export arrays retain the reference workbook's **17 question fields and 13 citation fields**, in snake_case. `gold_steps` is an array of `{step, result, unit, citations}` objects. Numeric answers are decimal strings.

Local checks verify required fields, citation joins, quotations against saved official text, units and the reviewer's arithmetic. **One independent reviewer per family** checks scenario feasibility, applicable law/year, answers and substantive variant diversity. Unknown rule-boundary dates may be null when applicability to the case is supported. Automatic approval is not human-expert certification.

After regular approval, a small separate agent creates missing-information extras by deleting a necessary numeric input and inheriting citations. These have `answer_value: "insufficient information"`, an explanation and gold steps. They receive local checks, with no second model review. Regular rows use `status: "validated"`; derived rows use `status: "ready"`. An invalid deletion gets one targeted repair using the same parent. If the extra still fails, it stays pending and the run continues with later families; approved regular exports are preserved. Resume retries pending extras without redoing regular work. Provider outages still pause the run.

Check `work/result.json` for completion, coverage and pending families. `work/families/` holds actionable feedback; `work/operations/` holds model/tool diagnostics. Unapproved drafts stay in `work/` and are not presented as validated dataset rows.

## Check the installation

```sh
.venv/bin/taxcalcbench doctor --country configs/kz.json --source-dir data/kz/temporal-sources
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/ruff check --no-cache .
```

`doctor --live` additionally makes a small paid SDK/tool check. Offline tests use synthetic fixtures, not invented tax-law evidence.

See [HPC preparation checks](docs/hpc-checks.md) for the downloaded source counts, live country-test results, measured API usage and remaining limits.

See the [architecture and implementation checklist](docs/architecture.md), [roadmap](docs/roadmap.md), and original materials in [references/](references/).

# Minimal tax-question creator

## Workflow

```mermaid
flowchart LR
    S[Download official sources once] --> P[Plan one family]
    P --> W[Write regular questions]
    W --> R[One independent review]
    R -->|Specific corrections| W
    R -->|Revise unsupported scenario| P
    R -->|Approved| M[Optional missing-information derivation]
    M --> E[Export and start next family]
```

OpenAI Agents SDK runs Gemini with separate planner, writer and reviewer contexts. Python handles official downloads, local files, arithmetic and checkpoints. There is one country and one language per run, with no database, experiment runner or runtime web search.

The multi-country CLI is a sequential loop around this same single-country function. Each country has separate source and output directories; no cross-country corpus, extra agent or database is introduced. The Slurm script invokes that CLI on one allocated workstation.

The country JSON sets official national sources, language, family size, OECD category weights and year-period weights. Downloads finish before generation; the corpus remains fixed throughout the run. An unsupported calculation must be redesigned using those files. A missing law is never replaced by memory or a fictional statutory value.

A country with a known unresolved source gap can set `generation.blocked_reason`. This prevents paid generation before it starts, while source downloads and diagnostics remain available; a multi-country batch reports the block and proceeds to the next country.

The default remains 50 families, five regular questions and one optional missing-information question per family. Regular cases balance the configured periods and categories and include a matched temporal pair. The extra case removes a necessary numeric input from an approved question and inherits its citations.

## Validation and recovery

Local checks bind citations to official source files, match quotations, check required fields and replay the reviewer's arithmetic. Temporal wording conflicts enter the same correction loop before review: both questions are revised to a shared scenario possible in both years, while other drafts are retained. One independent reviewer checks the scenario, applicable law/year, calculations and substantive variant diversity. Unknown rule-boundary dates may be null when applicability to the actual scenario is established. Formatting preferences and repeated variant labels are not reasons to reject sound questions.

Corrections use the saved draft and specific feedback without a user-managed correction counter. Repeated unchanged failures remain pending instead of spinning indefinitely. An invalid missing-information deletion gets a targeted edit repair; a remaining failure leaves that extra pending and does not stop later families. Temporary provider failures retry the same request using the SDK; a persistent failure saves progress and pauses before another family is started. Rerunning the same command resumes unfinished work and retains approvals.

A failed first plan does not lock a family to its original preferred category. Source-gap reports are diagnostic data; replanning may choose any category already permitted by the assignment, with a recorded reason. Resume checks saved approvals once before model work and skips finished families, rather than repeatedly scanning the whole dataset for each one.

A rendered question that exactly matches a known year template reuses that template locally. Other missing year placeholders are correction feedback, not fatal run errors; the code never guesses which numeric facts represent years.

A writer returning another assigned case gets targeted regeneration before its checkpoint is accepted. Native provider retries default to eight attempts after the original request, with visible progress and exponential backoff. Exhausted retries preserve the existing pause/resume behavior. Moving a run changes only its recorded filesystem location: evidence hashes and dataset settings still have to match.

Completed correction responses are recoverable if interruption occurs before the question checkpoint is written. A generic request to revise the family plan does not force the writer to regenerate unchanged cases; substantive feedback and changed case evidence still do. Interrupted downloads verify saved files and continue with unfinished URLs, while incomplete inventories remain unavailable to generation.

Extraction keeps original bytes alongside text. HTML merged cells retain explicit span information. Readable PDF pages survive unavailable cover/scanned/complex-graphics pages; gaps remain visible to agents and citation checks. Straight table rules are formatting. The optional Poppler backend handles verified RTL text extraction; scanned laws still require readable official alternatives. Country selectors may remove only explicitly identified decorative material. No tax rule is inferred by the downloader.

## Files

- `configs/<country>.json`: settings and official-source policy.
- `data/<country>/<collection>/`: originals, extracted text and source inventory.
- `outputs/<run>/questions.json`: the workbook's 17 question fields, in snake_case.
- `outputs/<run>/citations.json`: the workbook's 13 citation fields, in snake_case.
- `outputs/<run>/work/`: plans, drafts, reviews, progress and usage.

`gold_steps` is an array of `{step, result, unit, citations}` objects. Regular approved rows use `validated`; derived missing-information rows use `ready` and `answer_value: "insufficient information"`. Automatic approval is not human-expert certification.

## Implementation checklist

- [x] Enable native retries for temporary provider errors and prevent cascading failures.
- [x] Remove runtime source-acquisition tools; retain explicit official downloads.
- [x] Replace separate citation audits with one independent family review.
- [x] Complete each family in sequence, with automatic corrections and simple resume.
- [x] Remove correction-limit settings and obsolete code, tests and caches.
- [x] Rewrite the README around installation, settings and two run commands.
- [x] Test recovery, fixed-source behavior, review and export; run a focused live check.

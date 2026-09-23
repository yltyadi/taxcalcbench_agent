# HPC preparation checks — 22 September 2026

These checks used the local development machine and real Gemini calls. The code has **not yet run on an MBZUAI compute node**; its outbound network access and Slurm allocation policy still need the node-side check in the [README](../README.md#mbzuai-hpc--ubuntu).

## Official source preparation

All six configurations use fixed national official URLs with host/path allowlists. Generation reads the saved local laws; it has no web-search or download tool.

| Country | Language | Readable documents | Documents with explicitly excluded pages/provisions | Unavailable documents |
| --- | --- | ---: | ---: | ---: |
| Pakistan | English | 9 | 4 | 0 |
| India | English | 9 | 6 | 0 |
| China | Chinese | 15 | 1 | 0 |
| Indonesia | Indonesian | 13 | 1 | 0 |
| Poland | Polish | 16 | 0 | 0 |
| Egypt | Arabic | 2 | 0 | 13 |

The inventories preserve original bytes, extracted text, checksums and visible extraction gaps. These are starter source collections, not complete coverage of every tax and year. See the country-specific findings in [Pakistan/India](sources-pk-in.md), [China/Indonesia](sources-cn-id.md) and [Egypt/Poland](sources-eg-pl.md).

Egypt generation remains blocked by configuration, as requested. Its two readable recent laws cannot support the required historical periods. No OCR or unofficial replacement was added.

## Live one-family checks

Each successful check exports five independently reviewed regular questions and one derived missing-information question. The regular cases span the configured periods in a 3:1:1 distribution. These checks exercise one income-tax family per country; they do not establish complete coverage of all six OECD categories or certify legal correctness for publication.

| Country | Result | Exported questions | Model calls | Provider-reported tokens |
| --- | --- | ---: | ---: | ---: |
| Pakistan | Complete | 6 | 433 | 17,885,820 |
| India | Source gap; historical rate Act still needed | 0 | 87 | 3,832,418 |
| China | Complete | 6 | 117 | 4,159,945 |
| Indonesia | Complete | 6 | 169 | 5,927,531 |
| Poland | Complete | 6 | 262 | 10,905,093 |
| Egypt | Blocked before model calls | 0 | 0 | 0 |

Usage comes from each run's `work/result.json` and includes its saved retries/corrections. Failed requests without reported usage retain estimates in the separate accounted-token total. These numbers are not prices or a prediction for a full country run. The live checks are substantial API workloads despite producing only one family; inspect usage before scaling.

India's first planning attempt stopped because the source collection lacks a Finance Act establishing income-tax rates in the 2000s. The 2011 compilation does not replace the missing annual rate schedule. Additional official editions failed direct access or readable-text checks; a court archive also linked an Irish Act under an Indian-law index, which was rejected. No unsupported questions were exported. The [source report](sources-pk-in.md) records the remaining acquisition gap; India is blocked by configuration and excluded from the recommended full batch. Its 87 calls above are from the diagnostic run before this block.

The checked exports retain exactly 17 question fields and 13 citation fields, in snake_case, with structured gold steps. Regular rows are `validated`; the derived extra is `ready` with `answer_value: "insufficient information"`. All 139 citations across the four completed countries match their country/language inventory, official allowed URLs and exact passages in saved source text.

Artifacts are under `outputs/hpc-smoke/<code>/`; source collections are under `data/<code>/sources/`. Both roots are Git-ignored. Existing Kazakhstan data and output were preserved and excluded from these runs.

## Recovery and Linux checks

- **559 automated tests passed**, including moved-directory resume, source-integrity checks, transient-request recovery, targeted repair of a wrong assigned case, interrupted downloads, independent multi-country outputs, SIGTERM checkpoint preservation, source-gap replanning and completed-family resume.
- Ruff and shell syntax checks passed.
- A dependency dry run resolved all pinned requirements as Python 3.12 Linux x86_64 wheels. This verifies dependency availability, not execution on the actual HPC node.
- The real SDK/tool probe passed using `gemini-3.8-flash` through OpenAI Agents SDK.
- Live country runs encountered connection/time-out failures and continued through native SDK retries without manual restart.
- Completed Pakistan, China, Indonesia and Poland runs resumed together with both Gemini key environment variables empty, making no new model calls. Including Egypt and India reported `blocked`, retained the other completions and returned the expected partial-batch exit code 2.
- Standard interactive allocation (`salloc -N1 --mem=24G` inside tmux with the Conda environment) and subprocess SIGTERM tests verified graceful unwind and direct Python command execution. No wrapper scripts, GPU, database or additional scheduler framework are required.

The [README](../README.md) contains installation, one-country and multi-country commands, interactive HPC allocation and transfer/resume instructions. Full 50-family generation for the new countries has not been started.

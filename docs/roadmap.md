# Roadmap

The MVP creates a dataset for one country and one language from a fixed official corpus. Its [architecture](architecture.md) covers planning, writing, one independent review and optional missing-information derivation.

## Kazakhstan

- [x] Complete the 50-family run: 250 regular and 50 additional missing-information questions.
- [x] Check dataset-wide category breadth, regular-period targets and temporal contrasts in every family.
- [x] Resolve the remaining family through a supported planning choice in the existing corpus.
- [ ] Measure fresh family time and API usage with the simplified workflow.
- [ ] Perform publication-stage human validation.

KZ is excluded from the new-country batch. The local run completed on 22 September 2026 with **300 exported questions across 50 validated families** (250 reviewed regular + 50 missing-information cases). A-F16 replanned to a supported alternative in the same official corpus; all 294 previously exported questions and their citations were preserved unchanged. Regular cases meet the 150/50/50 period targets, all six categories are represented, and no families remain unresolved. These artifacts can remain archived locally rather than being transferred to HPC. Publication-stage human validation remains outstanding.

## Next countries

Use the single-language list in the [original guide](../references/Expert_Guide_and_Taxonomy.pdf). Each country needs its own official-source configuration, readable historical laws and one validated family before scaling. Tax rules stay in source files, not Python code.

| Country | Language |
| --- | --- |
| Pakistan | English (`en`) |
| India | English (`en`) |
| China | Chinese (`zh`) |
| Indonesia | Indonesian (`id`) |
| Poland | Polish (`pl`) |
| Egypt | Arabic (`ar`) |

Configurations and fixed official source downloads are implemented for all six countries. Pakistan, China, Indonesia and Poland each completed a live family with five reviewed regular questions and one missing-information extra; see [HPC checks](hpc-checks.md) for measured usage and limits. India's nine readable documents did not support a 2000s income-tax rate schedule, so its live check stopped without exporting unsupported questions. Generation is now blocked by configuration; India needs a readable, downloadable official historical Finance Act before the default income-tax family can proceed. The paired `sources-*.md` reports record actual editions and gaps.

Egypt generation is explicitly blocked by configuration at the user's request: two readable recent laws do not support the historical allocation. Resolving its official Arabic scans/corrupted text is future work, with no OCR or unofficial-source fallback added.

- [x] Portable resume using original source identities/checksums instead of machine paths.
- [x] Single-country commands and a sequential multi-country wrapper, with independent output folders.
- [x] MBZUAI Conda/Python 3.12 setup and direct execution in tmux/salloc.
- [x] Retry transient API failures with visible backoff; preserve progress after persistent outages.
- [ ] Extend official source packs where generation reports genuine legal dependencies.
- [ ] Scale each country after its small live check, then perform publication-stage human validation.

Kazakhstan remains Russian-only for this MVP. Bilingual generation, benchmark experiments and mixed-country/year retrieval corpora are future work outside this creator.

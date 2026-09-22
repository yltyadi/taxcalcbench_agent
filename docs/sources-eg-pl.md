# Egypt and Poland source preparation

Checked on 22 September 2026. Both configurations use fixed URLs and exact government host/path allowlists. There is no open-web search at generation time. The acquisition limits equal the number of configured documents; they are download bounds, not question limits. A successful download is not proof that every provision is readable or applicable to every requested year.

## Poland — Polish

`configs/pl.json` uses Poland's official [ELI service](https://eli.gov.pl/). All 16 configured HTML documents downloaded to `data/pl/sources`, with no extraction failures or withheld provisions after refreshing with the merged-table fix. The collection includes published or consolidated editions of:

- Personal income tax: 2000, 2010 and 2024.
- VAT: 2004, 2011 and 2024.
- Social insurance: 2009, 2015 and 2024.
- Civil-law transactions tax: 2005, 2015 and 2024.
- Inheritance and gift tax: 2009 and 2024, plus the 2023 amounts/scales regulation.
- Minimum wage for 2024.

These are selected editions, not a complete amendment history. The planner must verify commencement, later changes, and any required annual parameters in the available text. Civil-law transactions tax has readable material across all three periods. Income-tax Article 27 and inheritance-tax Article 15 now retain their tables, including merged-cell structure. The 2000 income-tax edition includes the applicable 2000 annual scale after Article 58, referenced by Article 27 footnote 187; use that expressly dated scale rather than the earlier base table printed within Article 27. The 2010 edition also includes expressly dated transitional scales for 2007 and 2008. The inheritance-tax regulation must be read alongside the statute because statutory base amounts and updated amounts can differ. The collection does not establish a separate payroll-tax or other-tax category merely because all six OECD categories appear in the configuration.

```sh
.venv/bin/taxcalcbench download --country configs/pl.json --source-dir data/pl/sources
```

## Egypt — Arabic: historical coverage remains blocked

`configs/eg.json` contains 15 fixed PDFs linked from the Egyptian Tax Authority's [income-tax catalogue](https://portal.eta.gov.eg/ar/content/qwanyn-aldrybt-ly-aldkhl), [VAT catalogue](https://portal.eta.gov.eg/ar/content/qwanyn-aldrybt-ly-alqymt-almdaft), and [other legislation catalogue](https://www.eta.gov.eg/en/node/837). The candidates cover income tax, VAT, property tax, stamp/development levies, procedures and small-business measures. They are national sources, not regional guidance.

The final download produced **2 readable documents and 13 extraction failures**. The readable documents are [Law 7/2024](https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.7-2024.pdf) and [Law 6/2025](https://portal.eta.gov.eg/sites/default/files/2025-02/law_no.6.of_.2025.pdf). This is insufficient for the configured historical periods or a complete income-tax corpus. Do not start a full multi-period Egypt dataset yet.

Egypt explicitly selects `sources.pdf_text_backend: "pdftotext"`. Install Poppler so `pdftotext` is on `PATH`; the program reports a missing-backend failure instead of silently using another extractor. It runs locally once per PDF, preserves page boundaries, and checks encoding, page count, output size and execution time. No OCR or model-generated transcription is used.

Observed limitations:

- Income-tax Law 91/2005 is an entirely scanned 71-page PDF. It has no extractable law text.
- Law 96/2015 and the VAT PDFs have broken font mappings, producing control characters and gibberish. Those pages are withheld.
- Law 26/2020 has legible text but embedded image watermarks; the conservative generic graphics check withholds its pages. Text-only Form objects are supported by Poppler; images and complex graphics are not transcribed.
- The authority's social-insurance Law 79/1975 PDF was excluded after both extractors reversed its visible identifiers to `97` and `5791`. An official host does not guarantee reliable text extraction.
- Poppler preserves Law 7/2024's observed year, amounts and commencement paragraph, whereas the default extractor reversed Arabic text or digits. This spot check does not establish that every Arabic PDF is safe.

To unblock historical generation, obtain readable official historical editions or add a separately verified OCR workflow. Do not substitute a filename year, inferred numeral correction, or a third-party consolidation for the missing legal text. WIPO's readable consolidated pages were investigated but not added: source-level government provenance was not established, and its regulation text includes third-party editorial links.

```sh
.venv/bin/taxcalcbench download --country configs/eg.json --source-dir data/eg/sources
```

Both configs pass the settings command. Source preparation and parser tests used no model API calls. Downloaded raw PDFs are retained alongside the inventory's failure records so extraction gaps remain inspectable.

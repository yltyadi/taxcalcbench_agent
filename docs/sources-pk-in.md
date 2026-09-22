# Pakistan and India source packs

These are fixed, national, English-language source lists for the same generic pipeline. They contain law text and supporting rules/notifications, with exact official URL-path allowlists and no link discovery. The generation defaults remain 50 families, five regular cases and one missing-information case per family; edit each country JSON to change them.

| Config | Official publishers | Included editions |
| --- | --- | --- |
| [`configs/pk.json`](../configs/pk.json) | Federal Board of Revenue | Income Tax Ordinance: October 2009, December 2016, July 2025; Sales Tax Act: July 2009, July 2015, June 2025; Federal Excise Act: October 2009, July 2017; enacted Finance Act 2022, including capital-value tax and customs schedules. |
| [`configs/in.json`](../configs/in.json) | Department of Revenue; GST Council | Income-tax Act/Finance Act 2011 compilation and Income-tax Rules 2011; enacted Finance Act 2020 including First Schedule; original CGST/IGST Acts 2017, CGST Amendment Act 2018, CGST Rules through June 2021, original goods rate notification 1/2017 and commencement notification 9/2017. |

The official catalogues are [FBR income-tax ordinances](https://www.fbr.gov.pk/Categ/Income-Tax-Ordinance/326), [FBR sales-tax acts](https://fbr.gov.pk/Categ/Sales-Tax-Act-1990/301/1000), [FBR finance acts](https://www.fbr.gov.pk/Categ/Finance-Acts/620), [DOR Income-tax Act and Rules 2011](https://dor.gov.in/income-tax-act-rules-2011), [GST Council legislation](https://gstcouncil.gov.in/central-gst), and [GST Council finance acts](https://gstcouncil.gov.in/finance-act). The config lists the actual document URLs; catalogue pages are not downloaded as laws.

## Run

```sh
.venv/bin/taxcalcbench download --country configs/pk.json --source-dir data/pk/sources
.venv/bin/taxcalcbench run --country configs/pk.json --source-dir data/pk/sources --output outputs/pk

.venv/bin/taxcalcbench download --country configs/in.json --source-dir data/in/sources
.venv/bin/taxcalcbench run --country configs/in.json --source-dir data/in/sources --output outputs/in
```

India explicitly uses `pdf_text_backend: pdftotext`, so install Poppler (`pdftotext` on PATH) on the computer/HPC node that downloads the sources. This is needed to preserve the DOR PDFs' nested text objects. Pakistan uses the default Python PDF backend. Source preparation uses no model API calls.

India generation is currently disabled by `generation.blocked_reason` after the live source-gap finding below. Its `download`, `settings` and `doctor` commands still work. The example `run` command becomes usable after the historical evidence is prepared and that setting is removed.

## Verified scope and limits

The source downloads and local extraction were checked on 22 September 2026. Pakistan produced 9 usable documents with no failed URLs; four documents retain explicitly withheld PDF pages. India produced 9 usable documents with no failed URLs; six documents retain explicitly withheld PDF pages. Withheld pages are represented as gaps, never silently offered as complete legal evidence. Raw PDFs and source hashes remain in each local source collection.

These packs are bounded starting corpora, not complete statements of each country's law for every year. A document's original enactment year, consolidation date and effective date of a provision are different facts. In particular:

- Pakistan has separate readable 2000s, 2010s and 2020s consolidations for income and sales taxes. Finance Act 2022 adds other federal provisions. Provincial taxes are excluded. No complete national social-insurance collection was verified.
- India has useful 2011 and 2020s national texts. The 2011 compilation contains quoted earlier provisions and dated amendments, but it does **not supply the annual income-tax rate schedules for 2000–2009**. A live planning check stopped on this source gap, so the current pack is not yet sufficient for the default temporal income-tax family. Current 2026 income-tax coverage and a full intervening amendment chain are also not supplied. The planner must choose only a year and calculation actually supported by the downloaded provisions.
- The DOR compilation includes publisher commentary. It is an official-hosted text, but commentary and case summaries must not be treated as enacted law. The included Finance Acts' statutory text and schedules are distinguishable from that commentary.
- GST rate notification 1/2017 is the original central goods schedule. It does not establish every later rate, State GST, IGST rates or service rates. A question must restrict its calculation to the component, period and provisions actually supplied.
- The six OECD categories are planning targets, not a claim that these packs provide every category. Missing categories or dependencies must be reallocated or reported, not filled from model memory.

Scanned Pakistan Finance Act 2015 and the oversized Finance Act 2025 candidate were omitted. Indian Labour Ministry candidates redirected to 404 pages; IndiaCode, CBDT and ESIC alternatives encountered timeout/access-denied responses from the test connection. Those failed candidates are not configured as usable sources. Finance Bills, press summaries and private-site reproductions were excluded even where their titles resembled enacted Acts.

Two official Gazette republications contain the missing national legislation: [Finance Act 2005, Act 18 of 2005](https://www.goaprintingpress.gov.in/downloads/0708/0708-27-SI-OG.pdf) and [Finance Act 2006](https://megpns.gov.in/gazette/2008/03/06-03-08-VII.pdf). Their contents include the First Schedule; their later Gazette republication dates do not change the provisions' effective dates. These are national Acts republished by state official printers, not regional tax provisions. They remain **unconfigured candidates**: direct acquisition encountered a TLS hostname mismatch/HTTP 403 for the Goa publisher and connection timeouts for Meghalaya. An indexed search result is not a substitute for acquiring and verifying the original source file.

The [Tripura High Court Central Acts index](https://thc.nic.in/listofcentralacts.html) was also checked. Downloads worked, but its 2005–2007 and 2009 Finance Act copies were scanned; the 2008 copy contained a poor OCR layer and was rejected by the normal extractor. Its readable 2001 file was explanatory commentary, and its 2002 file was **Ireland's Finance Act**, despite appearing in an Indian court's index. None was added. Official hosting alone does not establish a document's jurisdiction, legal status or extraction quality.

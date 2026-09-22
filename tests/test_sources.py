"""Synthetic HTTP and document fixtures only; no fabricated law text is official."""

import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from taxcalcbench import sources


def config():
    return {"country": "SYNTHETIC", "language": "xx", "sources": {
        "allowed_hosts": {"official.example": [r"/national/[a-z0-9._/-]+"]},
        "seeds": [{"url": "https://official.example/national/start.html", "title": "SYNTHETIC catalogue",
                   "kind": "navigation"}],
        "link_rules": [{"url_pattern": r"https://official\.example/national/(?:law|old)\.html",
                        "kind": "law", "priority": 5}],
        "section_heading_pattern": r"^Article [0-9]+\.", "retries": 0,
        "max_documents": 10, "max_pages": 20, "max_depth": 3}}


def serve(monkeypatch, responses):
    requests = []

    def handle(request):
        requests.append(str(request.url))
        response = responses[str(request.url)]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(request)
        if isinstance(response, bytes):
            return httpx.Response(200, content=response, headers={"content-type": "text/html"})
        return response

    monkeypatch.setattr(sources, "_make_client", lambda policy: httpx.Client(transport=httpx.MockTransport(handle)))
    return requests


def fixture_docx(body, *, extra=None, headers=None):
    """Build a minimal synthetic Word package, explicitly unrelated to tax law."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    parts = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f'<Override PartName="/word/document.xml" ContentType="{sources.MAIN_CONTENT_TYPE}"/></Types>',
        "_rels/.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="main" Type="{r}/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": f'<w:document xmlns:w="{w}" xmlns:r="{r}"><w:body>{body}</w:body></w:document>',
        **(extra or {}),
    }
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return output.getvalue()


def paragraph(text):
    return f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def synthetic_html(text="Article 1. SYNTHETIC\nA fictional rule for a parser test only."):
    return f"<html><body><article><p>{escape(text)}</p></article></body></html>".encode()


def test_observed_links_order_scope_provenance_and_immutable_cache(tmp_path, monkeypatch):
    cfg = config()
    start = cfg["sources"]["seeds"][0]["url"]
    law = "https://official.example/national/law.html"
    old = "https://official.example/national/old.html"
    calls = serve(monkeypatch, {start: b'<a href="old.html">Old synthetic</a><a href="law.html#x">Law</a>'
                               b'<a href="https://evil.example/national/law.html">Bad</a>'
                               b'<a href="/regional/law.html">Regional</a><a href="law.html#y">Duplicate</a>',
                               law: synthetic_html(), old: synthetic_html("Article 2. SYNTHETIC older")})
    inventory = sources.download_sources(cfg, tmp_path)
    assert calls == [start, law, old]
    assert len(inventory["documents"]) == 3
    child = inventory["documents"][1]
    assert child["observed_from"]["url"] == start
    assert child["observed_from"]["sha256"] == inventory["documents"][0]["sha256"]
    assert child["link_text"] == "Law"
    assert Path(child["raw_path"]).name == "law.html"
    assert child["boundaries"][0]["label"] == "Article 1. SYNTHETIC"
    assert sources.download_sources(cfg, tmp_path) == inventory
    assert calls == [start, law, old]


def test_unreadable_html_law_still_discovers_history_and_readable_edition(tmp_path, monkeypatch):
    cfg = config()
    law = "https://official.example/national/law.html"
    history = "https://official.example/national/history.html"
    old = "https://official.example/national/old.html"
    cfg["sources"]["seeds"] = [{"url": law, "title": "SYNTHETIC unreadable law", "kind": "law"}]
    cfg["sources"]["link_rules"].append({"url_pattern": r"https://official\.example/national/history\.html",
                                       "kind": "navigation", "priority": 1})
    raw = b'<article><img src="formula.png"><a href="history.html">Actual history link</a>'
    raw += b'<a href="https://evil.example/elsewhere">Outside</a><p>Article 1. Synthetic</p></article>'
    calls = serve(monkeypatch, {law: raw, history: b'<a href="old.html">Observed older edition</a>',
                               old: synthetic_html()})
    inventory = sources.download_sources(cfg, tmp_path)
    assert calls == [law, history, old]
    assert [row["url"] for row in inventory["documents"]] == [history, old]
    failed = inventory["failures"][0]
    assert failed["code"] == "UNISOLATED_EXTRACTION_GAP"
    parent = inventory["documents"][0]["observed_from"]
    assert parent == {"url": law, "sha256": failed["sha256"], "raw_path": failed["raw_path"]}
    assert (tmp_path / parent["raw_path"]).read_bytes() == raw
    assert inventory["documents"][0]["link_text"] == "Actual history link"


@pytest.mark.parametrize("mode,expected", [("non_html", "UNSUPPORTED_MEDIA_TYPE"),
                                            ("checksum_mismatch", "EXPECTED_CHECKSUM_MISMATCH")])
def test_rejected_response_cannot_supply_discovery_links(tmp_path, monkeypatch, mode, expected):
    cfg = config()
    start = cfg["sources"]["seeds"][0]["url"]
    if mode == "checksum_mismatch":
        cfg["sources"]["seeds"][0]["sha256"] = "0" * 64
    mime = "application/octet-stream" if mode == "non_html" else "text/html"
    calls = serve(monkeypatch, {start: httpx.Response(200, content=b'<a href="old.html">Never follow</a>',
                                                     headers={"content-type": mime})})
    inventory = sources.download_sources(cfg, tmp_path)
    assert calls == [start]
    assert inventory["failures"][0]["code"] == expected
    assert not inventory["documents"]


def test_redirect_is_checked_before_fetch(tmp_path, monkeypatch):
    cfg = config()
    start = cfg["sources"]["seeds"][0]["url"]
    calls = serve(monkeypatch, {start: httpx.Response(302, headers={"location": "https://evil.example/stolen"})})
    inventory = sources.download_sources(cfg, tmp_path)
    assert calls == [start]
    assert inventory["failures"][0]["code"] == "SOURCE_POLICY"
    assert inventory["documents"] == []


def test_approved_redirect_and_original_filename(tmp_path, monkeypatch):
    cfg = config()
    start = cfg["sources"]["seeds"][0]["url"]
    target = "https://official.example/national/law.html"
    calls = serve(monkeypatch, {start: httpx.Response(302, headers={"location": target}),
                               target: httpx.Response(200, content=synthetic_html(),
                                                     headers={"content-type": "text/html",
                                                              "content-disposition": 'attachment; filename="Original Name.html"'})})
    inventory = sources.download_sources(cfg, tmp_path)
    row = inventory["documents"][0]
    assert row["url"] == start and row["final_url"] == target
    assert row["redirects"] == calls == [start, target]
    assert Path(row["raw_path"]).name == "Original Name.html"


@pytest.mark.parametrize("url", [
    "http://official.example/national/law.html", "https://regional.official.example/national/law.html",
    "https://official.example.evil.example/national/law.html", "https://user:secret@official.example/national/law.html",
    "https://official.example:8443/national/law.html", "https://official.example/national/../regional/law.html",
    "https://official.example/national/%2e%2e/law.html", "https://official.example/national/%252e%252e/law.html",
    "https://official.example/national/law.html?country=elsewhere", "https://official.example/regional/law.html",
])
def test_exact_host_path_scheme_and_query_policy(url):
    with pytest.raises(sources.SourceError, match="policy|query"):
        sources._allowed(url, sources._policy(config()))


def test_retry_and_byte_limit_are_bounded(tmp_path, monkeypatch):
    cfg = config()
    cfg["sources"].update({"retries": 1, "max_bytes": 8})
    start = cfg["sources"]["seeds"][0]["url"]
    count = 0

    def response(request):
        nonlocal count
        count += 1
        return httpx.Response(503) if count == 1 else httpx.Response(200, content=b"0123456789",
                                                                   headers={"content-type": "text/plain"})

    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    calls = serve(monkeypatch, {start: response})
    inventory = sources.download_sources(cfg, tmp_path)
    assert len(calls) == 2
    assert inventory["failures"][0]["code"] == "DOWNLOAD_LIMIT"


def test_attempt_limit_counts_failed_law_sources(tmp_path, monkeypatch):
    cfg = config()
    cfg["sources"]["seeds"] = [{"url": "https://official.example/national/law.html", "title": "SYNTHETIC",
                                "kind": "law"},
                               {"url": "https://official.example/national/old.html", "title": "SYNTHETIC OLD",
                                "kind": "law"}]
    first = cfg["sources"]["seeds"][0]["url"]
    calls = serve(monkeypatch, {first: httpx.Response(404)})
    inventory = sources.download_sources(cfg, tmp_path, max_documents=1)
    assert calls == [first]
    assert inventory["documents_attempted"] == 1 and inventory["bounded"]
    assert [x["code"] for x in inventory["failures"]] == ["HTTP_STATUS", "DOCUMENT_LIMIT"]


@pytest.mark.parametrize("field,expected", [("max_depth", "DEPTH_LIMIT"), ("max_pages", "PAGE_LIMIT")])
def test_navigation_bounds(tmp_path, monkeypatch, field, expected):
    cfg = config()
    cfg["sources"][field] = 0 if field == "max_depth" else 1
    start = cfg["sources"]["seeds"][0]["url"]
    calls = serve(monkeypatch, {start: b'<a href="law.html">Synthetic law</a>'})
    inventory = sources.download_sources(cfg, tmp_path)
    assert calls == [start]
    assert inventory["failures"][0]["code"] == expected


def test_cache_tamper_policy_change_refresh_and_runtime_independence(tmp_path, monkeypatch):
    cfg = config()
    start = cfg["sources"]["seeds"][0]["url"]
    serve(monkeypatch, {start: synthetic_html()})
    inventory = sources.download_sources(cfg, tmp_path)
    runtime_change = {**cfg, "runtime": {"model": "different"}}
    assert sources.download_sources(runtime_change, tmp_path) == inventory
    changed = copy.deepcopy(cfg)
    changed["sources"]["max_depth"] = 2
    with pytest.raises(ValueError, match="policy changed"):
        sources.download_sources(changed, tmp_path)
    with pytest.raises(ValueError, match="new source folder"):
        sources.download_sources(cfg, tmp_path, refresh=True)
    text = tmp_path / inventory["documents"][0]["text_path"]
    text.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        sources.load_inventory(tmp_path)


def test_inventory_rejects_path_escape(tmp_path, monkeypatch):
    cfg = config()
    serve(monkeypatch, {cfg["sources"]["seeds"][0]["url"]: synthetic_html()})
    sources.download_sources(cfg, tmp_path)
    path = tmp_path / "sources.json"
    inventory = json.loads(path.read_text())
    inventory["documents"][0]["raw_path"] = "../outside"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="escapes"):
        sources.load_inventory(tmp_path)


def test_interrupted_inventory_cannot_be_used_as_prepared_sources(tmp_path, monkeypatch):
    cfg = config()
    serve(monkeypatch, {cfg["sources"]["seeds"][0]["url"]: synthetic_html()})
    sources.download_sources(cfg, tmp_path)
    path = tmp_path / "sources.json"
    inventory = json.loads(path.read_text())
    inventory["complete"] = False
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="interrupted"):
        sources.load_inventory(tmp_path)


def test_docx_preserves_text_table_cells_and_referenced_footnote():
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    rels += '<Relationship Id="notes" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/></Relationships>'
    body = paragraph("Article 1. SYNTHETIC") + '<w:tbl><w:tr><w:tc>' + paragraph("A")
    body += '</w:tc><w:tc>' + paragraph("B") + '</w:tc></w:tr></w:tbl>'
    body += '<w:p><w:r><w:footnoteReference w:id="2"/></w:r></w:p>'
    extra = {"word/_rels/document.xml.rels": rels,
             "word/footnotes.xml": '<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
             '<w:footnote w:id="2">' + paragraph("Synthetic footnote") + '</w:footnote></w:footnotes>'}
    result = sources._extract_docx(fixture_docx(body, extra=extra))
    assert "A\tB" in result.text and "[FOOTNOTE 2]" in result.text and "Synthetic footnote" in result.text


def test_style_inherited_list_numbering_is_not_silently_omitted():
    body = paragraph("Article 1. SYNTHETIC")
    body += '<w:p><w:pPr><w:pStyle w:val="child"/></w:pPr><w:r><w:t>Unnumbered display</w:t></w:r></w:p>'
    styles = '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    styles += '<w:style w:styleId="parent"><w:pPr><w:numPr/></w:pPr></w:style>'
    styles += '<w:style w:styleId="child"><w:basedOn w:val="parent"/></w:style></w:styles>'
    result = sources._extract(fixture_docx(body, extra={"word/styles.xml": styles}), sources.DOCX_MIME,
                              "https://official.example/national/law.docx", "law", sources._policy(config()))
    assert result["partial"] and "Unnumbered display" not in result["text"]


@pytest.mark.parametrize("component", ["<w:drawing/>", "<w:sym/>", "<w:numPr/>", "<w:gridSpan/>"])
def test_whole_affected_provision_is_withheld(component):
    body = paragraph("Article 1. SYNTHETIC unavailable") + paragraph("NEVER EXPOSE THIS RULE")
    body += '<w:p><w:r>' + component + '</w:r></w:p>' + paragraph("Article 2. SYNTHETIC readable")
    body += paragraph("Safe fictional test text")
    policy = sources._policy(config())
    result = sources._extract(fixture_docx(body), sources.DOCX_MIME,
                              "https://official.example/national/law.docx", "law", policy)
    assert result["partial"] and len(result["extraction_gaps"]) == 1
    assert "NEVER EXPOSE THIS RULE" not in result["text"]
    assert "Safe fictional test text" in result["text"]
    gap = result["extraction_gaps"][0]
    assert result["text"][gap["start"]:gap["end"]].startswith("[UNAVAILABLE PROVISION:")


def test_unisolated_drawing_rejects_and_retains_original(tmp_path, monkeypatch):
    cfg = config()
    url = "https://official.example/national/test.docx"
    cfg["sources"]["seeds"] = [{"url": url, "title": "SYNTHETIC DOCX", "kind": "law"}]
    raw = fixture_docx('<w:p><w:r><w:drawing/></w:r></w:p>' + paragraph("Article 1. Synthetic"))
    serve(monkeypatch, {url: httpx.Response(200, content=raw, headers={"content-type": sources.DOCX_MIME})})
    inventory = sources.download_sources(cfg, tmp_path)
    assert not inventory["documents"]
    failed = inventory["failures"][0]
    assert failed["code"] == "UNISOLATED_EXTRACTION_GAP"
    assert (tmp_path / failed["raw_path"]).read_bytes() == raw
    assert failed["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("extra,body,code", [
    ({"word/embeddings/oleObject1.bin": b"not executable"}, paragraph("Synthetic"), "DOCX_ACTIVE_CONTENT"),
    ({}, '<w:ins>' + paragraph("Synthetic") + '</w:ins>', "DOCX_TRACKED_CHANGES"),
    ({"../escape.xml": "<x/>"}, paragraph("Synthetic"), "DOCX_INVALID"),
    ({"word/extra.xml": '<!DOCTYPE x [<!ENTITY e "expanded">]><x>&e;</x>'}, paragraph("Synthetic"), "DOCX_XML_UNSAFE"),
])
def test_docx_unsafe_or_ambiguous_content_is_rejected(extra, body, code):
    with pytest.raises(sources.SourceError) as error:
        sources._extract_docx(fixture_docx(body, extra=extra))
    assert error.value.code == code


def test_html_images_withhold_complete_provision_and_selector_requires_actual_document():
    cfg = config()
    cfg["sources"]["html_selectors"] = {"official.example": "article"}
    policy = sources._policy(cfg)
    raw = b'<article><p>Article 1. Synthetic</p><p>Hidden rate <img src="formula.png"></p><p>Article 2. Visible</p><p>Test text</p></article>'
    result = sources._extract(raw, "text/html", "https://official.example/national/law.html", "law", policy)
    assert result["partial"] and "Hidden rate" not in result["text"] and "Test text" in result["text"]
    with pytest.raises(sources.SourceError, match="element is missing"):
        sources._extract(b"<html><body>Error page</body></html>", "text/html",
                         "https://official.example/national/law.html", "law", policy)


def test_text_pdf_preserves_rules_and_marks_blank_pages_unavailable():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (SYNTHETIC PDF TEST) Tj ET")
    page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    text, boundaries = sources._pdf(output.getvalue(), 10)
    assert "SYNTHETIC PDF TEST" in text and boundaries == [{"kind": "page", "label": "1", "start": 0, "end": len(text)}]
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (SYNTHETIC PDF TEST) Tj ET 0 0 m 5 5 l S")
    graphical = BytesIO()
    writer.write(graphical)
    assert "SYNTHETIC PDF TEST" in sources._pdf(graphical.getvalue(), 10)[0]
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (SYNTHETIC PDF TEST) Tj ET")
    writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    result = sources._extract(output.getvalue(), "application/pdf", "https://official.example/national/law.pdf",
                              "law", sources._policy(config()))
    assert result["partial"] and len(result["extraction_gaps"]) == 1
    assert result["extraction_gaps"][0]["label"] == "PDF page 2"
    assert "SYNTHETIC PDF TEST" in result["text"] and "UNAVAILABLE PDF PAGE 2" in result["text"]


def test_wholly_scanned_or_blank_pdf_is_not_a_readable_law():
    writer, output = PdfWriter(), BytesIO()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    with pytest.raises(sources.SourceError, match="no readable text pages"):
        sources._pdf(output.getvalue(), 10)


def synthetic_form_pdf(*, image=False, curves=False):
    writer, output = PdfWriter(), BytesIO()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    form = DecodedStreamObject()
    form[NameObject("/Subtype")] = NameObject("/Form")
    form[NameObject("/Resources")] = resources
    form.set_data(b"BT /F1 12 Tf 72 720 Td (SYNTHETIC FORM) Tj ET" +
                  (b" 0 0 1 1 2 2 c S" if curves else b""))
    if image:
        raster = DecodedStreamObject()
        raster[NameObject("/Subtype")] = NameObject("/Image")
        resources[NameObject("/XObject")] = DictionaryObject({NameObject("/Im0"): raster})
        form.set_data(b"/Im0 Do")
    # Exercise an indirect resource dictionary, as in real government PDFs.
    page[NameObject("/Resources")] = writer._add_object(DictionaryObject({
        NameObject("/XObject"): writer._add_object(DictionaryObject({NameObject("/Fm0"): form}))}))
    stream = DecodedStreamObject()
    stream.set_data(b"/Fm0 Do")
    page[NameObject("/Contents")] = stream
    writer.write(output)
    return output.getvalue()


def test_pdf_text_forms_are_allowed_but_nested_images_and_curves_are_withheld(monkeypatch):
    monkeypatch.setattr(sources, "_pdftotext_pages", lambda raw, count: ["SYNTHETIC FORM"])
    assert "SYNTHETIC FORM" in sources._pdf(synthetic_form_pdf(), 10, "pdftotext")[0]
    # pypdf layout extraction omits Form text; do not offer incomplete evidence.
    with pytest.raises(sources.SourceError, match="no readable text pages"):
        sources._pdf(synthetic_form_pdf(), 10)
    for raw in (synthetic_form_pdf(image=True), synthetic_form_pdf(curves=True)):
        with pytest.raises(sources.SourceError, match="no readable text pages"):
            sources._pdf(raw, 10, "pdftotext")


def test_explicit_pdftotext_backend_preserves_rtl_and_page_boundaries(monkeypatch):
    calls = []
    arabic = "قانون رقم ٧ لسنة ٢٠٢٤\nمبلغ ٦٠٠,٠٠٠ جنيه\n"

    def run(command, **kwargs):
        calls.append(command)
        assert command[:4] == ["pdftotext", "-layout", "-enc", "UTF-8"]
        assert kwargs["timeout"] == 60 and kwargs["check"]
        assert Path(command[-2]).read_bytes().startswith(b"%PDF-")
        Path(command[-1]).write_text(arabic + "\f", encoding="utf-8")

    monkeypatch.setattr(sources.subprocess, "run", run)
    cfg = config()
    cfg["sources"]["pdf_text_backend"] = "pdftotext"
    result = sources._extract(synthetic_form_pdf(), "application/pdf",
        "https://official.example/national/law.pdf", "law", sources._policy(cfg))
    assert arabic in result["text"] and not result["partial"]
    assert len(calls) == 1 and len(result["boundaries"]) == 1
    assert "PDF_TEXT_BACKEND: pdftotext" in result["extraction_warnings"]


@pytest.mark.parametrize("failure,code", [
    (FileNotFoundError(), "PDF_BACKEND_MISSING"),
    (sources.subprocess.TimeoutExpired("pdftotext", 60), "PDF_EXTRACTION_TIMEOUT"),
    (sources.subprocess.CalledProcessError(1, "pdftotext"), "PDF_EXTRACTION_FAILED"),
])
def test_explicit_pdf_backend_does_not_silently_fallback(monkeypatch, failure, code):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(sources.subprocess, "run", run)
    with pytest.raises(sources.SourceError) as error:
        sources._pdf(synthetic_form_pdf(), 10, "pdftotext")
    assert error.value.code == code


@pytest.mark.parametrize("text,code", [(b"one\ftwo\f", "PDF_PAGE_MISMATCH"), (b"\xff\f", "TEXT_DECODING")])
def test_pdf_backend_rejects_invalid_page_count_or_encoding(monkeypatch, text, code):
    monkeypatch.setattr(sources.subprocess, "run", lambda command, **kwargs: Path(command[-1]).write_bytes(text))
    with pytest.raises(sources.SourceError) as error:
        sources._pdf(synthetic_form_pdf(), 10, "pdftotext")
    assert error.value.code == code


def test_pdf_backend_rejects_oversized_output(monkeypatch):
    def run(command, **kwargs):
        with Path(command[-1]).open("wb") as handle:
            handle.truncate(32 * 1024 * 1024 + 1)
    monkeypatch.setattr(sources.subprocess, "run", run)
    with pytest.raises(sources.SourceError) as error:
        sources._pdf(synthetic_form_pdf(), 10, "pdftotext")
    assert error.value.code == "TEXT_LIMIT"


def test_pdf_with_corrupt_font_mapping_is_not_readable_text(monkeypatch):
    monkeypatch.setattr(sources, "_pdftotext_pages", lambda raw, count: ["Synthetic broken \x1f glyph"])
    with pytest.raises(sources.SourceError, match="no readable text pages"):
        sources._pdf(synthetic_form_pdf(), 10, "pdftotext")


def test_pdf_backend_configuration_is_explicit():
    cfg = config()
    cfg["sources"]["pdf_text_backend"] = "invented"
    with pytest.raises(ValueError, match="backend"):
        sources._policy(cfg)


def test_html_keeps_merged_tax_table_structure_and_only_explicitly_excludes_logo():
    cfg = config()
    cfg["sources"]["html_remove_selectors"] = {"official.example": ["img#official-seal"]}
    raw = b'<article><img id="official-seal" src="seal.png"><p>Article 1. Bands</p><table><tr><th colspan="2">Tax</th></tr><tr><td rowspan="2">100</td><td>5%</td></tr><tr><td>10%</td></tr></table><p>Article 2. Formula</p><img src="formula.png"></article>'
    result = sources._extract(raw, "text/html", "https://official.example/national/law.html", "law", sources._policy(cfg))
    assert '<th colspan="2" rowspan="1">Tax</th>' in result["text"]
    assert '<td colspan="1" rowspan="2">100</td>' in result["text"]
    assert "5%" in result["text"] and "10%" in result["text"]
    assert result["partial"] and len(result["extraction_gaps"]) == 1


def test_active_country_sources_remain_national_russian_and_exactly_bound():
    cfg = json.loads((Path(__file__).parents[1] / "configs/kz.json").read_text())
    policy = sources._policy(cfg)
    assert cfg["country"] == "KZ" and cfg["language"] == "ru"
    for forbidden in ["https://zhmb.kgd.gov.kz/ru/content/nalogovyy-kodeks-rk",
                      "https://www.gov.kz/uploads/unrelated.docx",
                      "https://adilet.zan.kz/kaz/docs/K2500000214",
                      "https://adilet.zan.kz/rus/docs/V23A0001234"]:
        with pytest.raises(sources.SourceError):
            sources._allowed(forbidden, policy)
    assert policy["seeds"][0]["url"].endswith("1690294.docx")

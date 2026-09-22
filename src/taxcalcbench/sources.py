"""Download configured official sources and keep original bytes plus bounded text.

There is no legal applicability inference here. Source dates, filenames and link
labels are descriptive metadata. Image/formula/layout material we cannot preserve
is withheld at a configured provision boundary, or the document is rejected.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import ssl
import stat
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import httpx
import truststore
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader
from pypdf.generic import ContentStream

GAP_MARKER = "[UNREADABLE_SOURCE_COMPONENT]"
INVENTORY_VERSION = 1

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MAIN_CONTENT_TYPE = DOCX_MIME + ".main+xml"
WORD_NAMESPACES = {
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
}
REL_NAMESPACES = {
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "http://purl.oclc.org/ooxml/officeDocument/relationships",
}


class SourceError(ValueError):
    """A controlled acquisition/extraction failure without remote response text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class _Extraction:
    text: str
    warnings: tuple[str, ...] = ()


def _fail(code: str, message: str) -> None:
    raise SourceError(code, message)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _word(element: ET.Element, name: str) -> bool:
    return any(element.tag == f"{{{namespace}}}{name}" for namespace in WORD_NAMESPACES)


def _attribute(element: ET.Element, name: str, namespaces=WORD_NAMESPACES) -> str | None:
    for namespace in namespaces:
        value = element.get(f"{{{namespace}}}{name}")
        if value is not None:
            return value
    return None


class _BoundedBuilder(ET.TreeBuilder):
    def __init__(self, budget: list[int], max_elements: int, max_depth: int):
        super().__init__()
        self.budget, self.max_elements, self.max_depth = budget, max_elements, max_depth
        self.depth = 0

    def start(self, tag, attrs):
        self.budget[0] += 1
        self.depth += 1
        if self.budget[0] > self.max_elements or self.depth > self.max_depth:
            _fail("DOCX_XML_LIMIT", "DOCX exceeds the configured XML element or depth limit")
        return super().start(tag, attrs)

    def end(self, tag):
        self.depth -= 1
        return super().end(tag)

    def doctype(self, name, pubid, system):
        _fail("DOCX_XML_UNSAFE", "DOCX XML document type declarations are not supported")


def _relationship_owner(name: str) -> str:
    if name == "_rels/.rels":
        return ""
    directory, filename = posixpath.split(name)
    if posixpath.basename(directory) != "_rels":
        _fail("DOCX_INVALID", "DOCX relationship part has an unsupported location")
    return posixpath.join(posixpath.dirname(directory), filename[:-5])


def _target(owner: str, target: str) -> str:
    if not target or "\\" in target or any(char in target for char in ("\x00", "?", "#", "%", ":")):
        _fail("DOCX_INVALID", "DOCX contains an unsupported internal relationship target")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(owner), target)) if not target.startswith("/") else posixpath.normpath(target[1:])
    if resolved in ("", ".", "..") or resolved.startswith("../"):
        _fail("DOCX_INVALID", "DOCX internal relationship escapes its package")
    return resolved


def _resolve_alternate_content(tree: ET.Element, warnings: set[str]) -> None:
    """Select fallback only when every branch has identical visible text structure.

    Drawing wrappers differ between modern Word and VML, so compare visible text,
    paragraph/table boundaries, explicit breaks and note references. This does not
    establish image or layout equivalence; those remain explicit extraction gaps.
    """
    namespace = "http://schemas.openxmlformats.org/markup-compatibility/2006"

    def signature(element: ET.Element) -> tuple:
        local = _local(element.tag)
        if local == "t":
            return (("text", element.text or ""),)
        if local == "textpath":
            return (("textpath", element.get("string", "")),)
        word = any(element.tag.startswith(f"{{{item}}}") for item in WORD_NAMESPACES)
        if word and local in {"tab", "br", "cr", "noBreakHyphen", "softHyphen"}:
            return ((local,),)
        if word and local in {"footnoteReference", "endnoteReference"}:
            return ((local, _attribute(element, "id")),)
        pieces = tuple(item for child in element for item in signature(child))
        if word and local in {"p", "tbl", "tr", "tc"}:
            return (("start", local), *pieces, ("end", local))
        return pieces

    def resolve(element: ET.Element):
        for child in list(element):
            resolve(child)
        if _local(element.tag) != "AlternateContent":
            return
        branches = list(element)
        choices = [branch for branch in branches if branch.tag == f"{{{namespace}}}Choice"]
        fallbacks = [branch for branch in branches if branch.tag == f"{{{namespace}}}Fallback"]
        if (element.tag != f"{{{namespace}}}AlternateContent" or not choices or len(fallbacks) > 1
                or len(choices) + len(fallbacks) != len(branches)):
            _fail("DOCX_AMBIGUOUS_CONTENT", "DOCX alternate rendering branches are missing or unsupported")
        signatures = [signature(branch) for branch in branches]
        if any(item != signatures[0] for item in signatures[1:]):
            _fail("DOCX_AMBIGUOUS_CONTENT", "DOCX alternate rendering branches contain different visible text")
        selected = fallbacks[0] if fallbacks else choices[0]
        element[:] = list(selected)
        warnings.add("ALTERNATE_CONTENT_IDENTICAL_TEXT_BRANCH_SELECTED")
        warnings.add("ALTERNATE_CONTENT_VISUAL_EQUIVALENCE_UNQUALIFIED")

    resolve(tree)


def _extract_docx(
    raw: bytes, *, max_input_bytes: int = 20 * 1024 * 1024,
    max_uncompressed_bytes: int = 80 * 1024 * 1024,
    max_part_bytes: int = 32 * 1024 * 1024,
    max_text_bytes: int = 32 * 1024 * 1024,
    max_entries: int = 2000, max_xml_elements: int = 500_000,
    max_xml_depth: int = 128,
) -> _Extraction:
    """Read visible XML text; retain table rows/cells with LF/TAB separators.

Output order is unique referenced headers, main body, referenced footnotes and
endnotes, then unique referenced footers. Supplementary sections and references
have explicit extraction labels, not source page numbers. Numbering derived from
Word styles is not synthesized. No ZIP member is written or relationship fetched.
"""
    if not isinstance(raw, bytes):
        raise TypeError("DOCX input must be bytes")
    limits = (max_input_bytes, max_uncompressed_bytes, max_part_bytes, max_text_bytes,
              max_entries, max_xml_elements, max_xml_depth)
    if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError("DOCX extraction limits must be positive integers")
    if len(raw) > max_input_bytes:
        _fail("DOCX_INPUT_LIMIT", "DOCX exceeds the configured input byte limit")
    warnings = {"LAYOUT_AND_LEGAL_MEANING_UNQUALIFIED", "EXTRACTION_ORDER_HEADERS_BODY_NOTES_FOOTERS"}
    try:
        with ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                _fail("DOCX_PACKAGE_LIMIT", "DOCX exceeds the configured ZIP entry limit")
            names: set[str] = set()
            total = 0
            xml: dict[str, ET.Element] = {}
            xml_entries = []
            budget = [0]
            for entry in entries:
                name = entry.filename
                parts = name.rstrip("/").split("/")
                if (not name or name.startswith("/") or "\\" in name or "\x00" in name
                        or any(part in ("", ".", "..") for part in parts)
                        or name.casefold() in names or stat.S_ISLNK(entry.external_attr >> 16)):
                    _fail("DOCX_INVALID", "DOCX ZIP members have unsafe or ambiguous names")
                names.add(name.casefold())
                if entry.flag_bits & 1:
                    _fail("DOCX_ENCRYPTED", "Encrypted DOCX ZIP members are not supported")
                if entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                    _fail("DOCX_UNSUPPORTED_COMPRESSION", "DOCX uses unsupported ZIP compression")
                total += entry.file_size
                if entry.file_size > max_part_bytes or total > max_uncompressed_bytes:
                    _fail("DOCX_PACKAGE_LIMIT", "DOCX exceeds the configured uncompressed byte limit")
                lowered = name.lower()
                if any(marker in lowered for marker in ("vbaproject", "/activex/", "/embeddings/")):
                    _fail("DOCX_ACTIVE_CONTENT", "DOCX contains macros or active embedded content")
                if lowered.endswith((".xml", ".rels")):
                    xml_entries.append(entry)
            # Validate the complete directory before inflating any XML member.
            for entry in xml_entries:
                with archive.open(entry) as member:
                    data = member.read(max_part_bytes + 1)
                if len(data) > max_part_bytes or len(data) != entry.file_size:
                    _fail("DOCX_PACKAGE_LIMIT", "DOCX XML part exceeds its declared or configured size")
                parser = ET.XMLParser(target=_BoundedBuilder(budget, max_xml_elements, max_xml_depth))
                xml[entry.filename] = ET.fromstring(data, parser=parser)

            content_types = xml.get("[Content_Types].xml")
            if content_types is None:
                _fail("DOCX_INVALID", "DOCX content types are missing")
            main_parts = []
            for item in content_types:
                content_type = item.get("ContentType", "")
                if any(marker in content_type.lower() for marker in ("macroenabled", "vbaproject", "activex", "oleobject")):
                    _fail("DOCX_ACTIVE_CONTENT", "Macro-enabled or active DOCX content is not supported")
                if content_type == MAIN_CONTENT_TYPE:
                    main_parts.append(item.get("PartName", "").lstrip("/"))
            if len(main_parts) != 1 or main_parts[0] not in xml:
                _fail("DOCX_INVALID", "DOCX must declare exactly one supported main document")
            main_name = main_parts[0]
            relationships: dict[str, dict[str, tuple[str, str | None]]] = {}
            for name, tree in xml.items():
                if not name.endswith(".rels"):
                    continue
                owner = _relationship_owner(name)
                mapping: dict[str, tuple[str, str | None]] = {}
                for relation in tree:
                    kind = relation.get("Type", "").rsplit("/", 1)[-1]
                    ident, target = relation.get("Id", ""), relation.get("Target", "")
                    if not ident or ident in mapping:
                        _fail("DOCX_INVALID", "DOCX relationship identifiers are missing or duplicated")
                    if kind in {"attachedTemplate", "oleObject", "package", "control", "aFChunk"}:
                        _fail("DOCX_ACTIVE_CONTENT", "DOCX contains an active embedded or linked object")
                    mode = relation.get("TargetMode", "Internal")
                    if mode == "External":
                        if kind == "hyperlink":
                            warnings.add("EXTERNAL_HYPERLINKS_NOT_FOLLOWED")
                        elif kind == "image":
                            warnings.add("EXTERNAL_IMAGES_NOT_RETRIEVED")
                        else:
                            _fail("DOCX_ACTIVE_CONTENT", "DOCX contains an unsupported external relationship")
                        mapping[ident] = (kind, None)
                    elif mode == "Internal":
                        resolved = _target(owner, target)
                        if resolved.casefold() not in names:
                            _fail("DOCX_INVALID", "DOCX relationship points to a missing package part")
                        mapping[ident] = (kind, resolved)
                    else:
                        _fail("DOCX_INVALID", "DOCX contains an unsupported relationship mode")
                relationships[owner] = mapping
            main_links = [target for kind, target in relationships.get("", {}).values() if kind == "officeDocument"]
            if main_links != [main_name]:
                _fail("DOCX_INVALID", "DOCX package relationship does not identify its main document")

            tracked = {"ins", "del", "moveFrom", "moveTo", "moveFromRangeStart", "moveFromRangeEnd",
                       "moveToRangeStart", "moveToRangeEnd", "cellIns", "cellDel", "cellMerge"}
            for tree in xml.values():
                # Field instructions may be split between runs. Joining the instruction
                # fragments catches an active opcode without using its cached display.
                instructions = "".join(
                    element.text or "" for element in tree.iter() if _word(element, "instrText")
                )
                if re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK|IMPORT|DATABASE)\b", instructions, re.IGNORECASE):
                    _fail("DOCX_ACTIVE_CONTENT", "DOCX contains an external or active field instruction")
                for element in tree.iter():
                    local = _local(element.tag)
                    if any(element.tag.startswith(f"{{{namespace}}}") for namespace in WORD_NAMESPACES):
                        if local in tracked or local.endswith("PrChange") or local.startswith(("customXmlInsRange", "customXmlDelRange", "customXmlMove")):
                            _fail("DOCX_TRACKED_CHANGES", "DOCX contains unresolved tracked changes")
                        if local in {"altChunk", "object", "control"}:
                            _fail("DOCX_ACTIVE_CONTENT", "DOCX contains active or externally rendered content")
                        if local in {"vanish", "webHidden"} and _attribute(element, "val") not in {"0", "false", "off"}:
                            _fail("DOCX_HIDDEN_TEXT", "DOCX contains hidden text requiring separate review")
                        if local in {"numPr", "numStyleLink"}:
                            warnings.add("STYLE_DERIVED_NUMBERING_NOT_RENDERED")
                        if local in {"drawing", "pict", "txbxContent"}:
                            warnings.add("DRAWING_LAYOUT_AND_IMAGES_NOT_RENDERED")
                        if local == "sym":
                            warnings.add("FONT_SYMBOL_GLYPHS_NOT_DECODED")
                        if local in {"gridSpan", "vMerge", "hMerge"}:
                            warnings.add("MERGED_TABLE_CELLS_REQUIRE_REVIEW")
                        if local in {"instrText", "fldSimple"}:
                            warnings.add("FIELD_RESULTS_NOT_RECALCULATED")
                            instruction = element.text or "" if local == "instrText" else _attribute(element, "instr") or ""
                            if re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK|IMPORT|DATABASE)\b", instruction, re.IGNORECASE):
                                _fail("DOCX_ACTIVE_CONTENT", "DOCX contains an external or active field instruction")
                    if local == "t" and not _word(element, "t"):
                        warnings.add("NON_WORD_DRAWING_TEXT_NOT_EXTRACTED")

            # Inspect every branch for unsafe content above before selecting any
            # alternative. Discarded markup cannot conceal tracked or active data.
            for tree in xml.values():
                _resolve_alternate_content(tree, warnings)

            return _extract_parts(xml, relationships, main_name, warnings, max_text_bytes)
    except SourceError:
        raise
    except (BadZipFile, ET.ParseError, KeyError, ValueError, OSError, RuntimeError, RecursionError, NotImplementedError) as exc:
        raise SourceError("DOCX_INVALID", "DOCX package or XML could not be parsed reliably") from exc


def _extract_parts(xml, relationships, main_name, warnings, max_text_bytes) -> _Extraction:
    note_references: dict[str, list[str]] = {"footnote": [], "endnote": []}
    styles = { _attribute(element, "styleId"): element for tree in xml.values()
               if _word(tree, "styles") for element in tree if _word(element, "style") }
    default_styles = [ident for ident, element in styles.items()
                      if _attribute(element, "type") == "paragraph" and _attribute(element, "default") == "1"]

    def numbered_style(ident: str | None, visited=None) -> bool:
        visited = set() if visited is None else visited
        if ident is None or ident not in styles:
            return False
        if ident in visited:
            _fail("DOCX_INVALID", "DOCX paragraph styles have cyclic inheritance")
        visited.add(ident)
        style = styles[ident]
        if any(_word(element, "numPr") for element in style.iter()):
            return True
        parent = next((_attribute(element, "val") for element in style if _word(element, "basedOn")), None)
        return numbered_style(parent, visited)

    numbered = {ident for ident in styles if numbered_style(ident)}

    def render(element: ET.Element) -> str:
        local = _local(element.tag)
        if _word(element, "p"):
            selected = [_attribute(child, "val") for properties in element if _word(properties, "pPr")
                        for child in properties if _word(child, "pStyle")] or default_styles
            if any(ident in numbered for ident in selected):
                return GAP_MARKER + "\n"
        # Do not silently omit formula images, unsupported mathematical markup,
        # font symbols, numbering, or table geometry. Their whole provision is
        # withheld by _withhold_gaps, using country-configured heading syntax.
        if local in {"drawing", "pict", "sym", "oMath", "oMathPara", "numPr",
                     "gridSpan", "vMerge", "hMerge"}:
            return GAP_MARKER
        if local == "t" and not _word(element, "t"):
            return GAP_MARKER
        if _word(element, "t"):
            return element.text or ""
        if _word(element, "tab"):
            return "\t"
        if _word(element, "br") or _word(element, "cr"):
            return "\n"
        if _word(element, "noBreakHyphen"):
            return "\u2011"
        if _word(element, "softHyphen"):
            return "\u00ad"
        if local in {"footnoteReference", "endnoteReference"} and _word(element, local):
            kind = local.removesuffix("Reference")
            ident = _attribute(element, "id")
            if ident is None or not re.fullmatch(r"-?\d+", ident):
                _fail("DOCX_INVALID", "DOCX note reference lacks a supported identifier")
            if ident not in note_references[kind]:
                note_references[kind].append(ident)
            return f"[{kind.upper()} {ident}]"
        pieces = [render(child) for child in element]
        if _word(element, "tr"):
            return "\t".join(piece.rstrip("\n") for child, piece in zip(element, pieces, strict=True) if _word(child, "tc")) + "\n"
        text = "".join(pieces)
        if _word(element, "p"):
            return text + "\n"
        return text

    main = xml[main_name]
    if not _word(main, "document"):
        _fail("DOCX_INVALID", "DOCX main XML is not a supported Word document")
    bodies = [child for child in main if _word(child, "body")]
    if len(bodies) != 1:
        _fail("DOCX_INVALID", "DOCX main document must contain one body")
    sections: dict[str, list[str]] = {"header": [], "footer": []}
    for element in main.iter():
        for kind, section_names in sections.items():
            if _word(element, kind + "Reference"):
                ident = _attribute(element, "id", REL_NAMESPACES)
                relation_kind, target = relationships.get(main_name, {}).get(ident, (None, None))
                if relation_kind != kind or target not in xml:
                    _fail("DOCX_INVALID", "DOCX header or footer reference is unresolved")
                if target not in section_names:
                    section_names.append(target)
    output: list[str] = []
    output_size = 0

    def append(text: str):
        nonlocal output_size
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        output_size += len(text.encode("utf-8"))
        if output_size > max_text_bytes:
            _fail("DOCX_TEXT_LIMIT", "DOCX exceeds the configured extracted UTF-8 byte limit")
        output.append(text)

    for index, name in enumerate(sections["header"], 1):
        append(f"[EXTRACTED HEADER {index}]\n" + render(xml[name]))
    body_text = render(bodies[0])
    if not body_text.strip():
        _fail("DOCX_NO_EXTRACTABLE_TEXT", "DOCX body contains no extractable text")
    append(body_text)
    # References in footers are registered before resolving notes; footers remain last.
    footers = [render(xml[name]) for name in sections["footer"]]
    for kind, references in note_references.items():
        targets = [target for rel_kind, target in relationships.get(main_name, {}).values() if rel_kind == kind + "s"]
        if len(targets) > 1 or (references and (len(targets) != 1 or targets[0] not in xml)):
            _fail("DOCX_INVALID", "DOCX note part is missing or ambiguous")
        notes: dict[str, ET.Element] = {}
        if targets:
            for note in xml[targets[0]]:
                if not _word(note, kind) or _attribute(note, "type") in {"separator", "continuationSeparator", "continuationNotice"}:
                    continue
                ident = _attribute(note, "id")
                if ident is None or ident in notes:
                    _fail("DOCX_INVALID", "DOCX note identifiers are missing or duplicated")
                notes[ident] = note
        for ident in references:
            if ident not in notes:
                _fail("DOCX_INVALID", "DOCX references a missing note")
            append(f"[EXTRACTED {kind.upper()} {ident}]\n" + render(notes[ident]))
        if set(notes) - set(references):
            warnings.add("UNREFERENCED_NOTES_NOT_INCLUDED")
    for index, text in enumerate(footers, 1):
        append(f"[EXTRACTED FOOTER {index}]\n" + text)
    if sections["header"] or sections["footer"]:
        warnings.add("HEADER_FOOTER_ORDER_NOT_PAGINATED")
    if any(_word(element, "tbl") for tree in xml.values() for element in tree.iter()):
        warnings.add("TABLE_ROW_CELL_ORDER_NOT_VISUAL_LAYOUT")
    text = "".join(output)
    if "\ufffd" in text:
        _fail("TEXT_DECODING", "DOCX contains replacement characters")
    return _Extraction(text=text, warnings=tuple(sorted(warnings)))


def _withhold_gaps(text: str, pattern: str | None) -> tuple[str, list[dict], list[dict]]:
    """Remove complete affected provisions; offsets refer to the resulting text."""
    headings = list(re.finditer(pattern, text, re.MULTILINE)) if pattern else []
    if GAP_MARKER in text and (not headings or GAP_MARKER in text[:headings[0].start()]):
        _fail("UNISOLATED_EXTRACTION_GAP", "Unreadable material cannot be isolated to a provision")
    output = text[:headings[0].start()] if headings else text
    gaps, boundaries = [], []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.start():end]
        label = block.splitlines()[0].strip()
        if GAP_MARKER in block:
            # Notes/headers are separate text parts: do not incorrectly assign
            # an unreadable endnote to whichever article happens to precede it.
            supplement = block.find("[EXTRACTED ")
            if supplement >= 0 and GAP_MARKER in block[supplement:]:
                _fail("UNISOLATED_EXTRACTION_GAP", "Unreadable supplementary content is not article-bound")
            block = f"[UNAVAILABLE PROVISION: {label}; unsupported image, formula, or layout]\n"
            gaps.append({"label": label, "reason": "UNSUPPORTED_NON_TEXT_OR_LAYOUT",
                         "start": len(output), "end": len(output) + len(block)})
        boundaries.append({"kind": "section", "label": label,
                           "start": len(output), "end": len(output) + len(block)})
        output += block
    return output, gaps, boundaries


def _html(raw: bytes, selector: str | None, remove_selectors: list[str] | None = None) -> tuple[str, str, list[tuple[str, str]]]:
    soup = BeautifulSoup(raw, "html.parser")
    links = [(tag.get("href"), tag.get_text(" ", strip=True)) for tag in soup.find_all("a", href=True)]
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    body = soup.select_one(selector) if selector else (soup.body or soup)
    if body is None:
        _fail("HTML_CONTENT_MISSING", "The configured official document element is missing")
    for excluded in remove_selectors or []:
        for element in body.select(excluded):
            element.decompose()
    for element in body.select("script, style, nav, noscript, template, form"):
        element.decompose()

    def render(node):
        if isinstance(node, NavigableString):
            return str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name in {"img", "svg", "canvas", "math", "object", "iframe", "embed"}:
            return GAP_MARKER
        if node.has_attr("hidden") or re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
                                               node.get("style", ""), re.IGNORECASE):
            return GAP_MARKER if node.get_text(strip=True) else ""
        if node.name == "br":
            return "\n"
        children = [render(child) for child in node.children]
        # Preserve merged-cell structure explicitly instead of dropping an
        # entire provision for a table heading or tax-rate band spanning cells.
        if node.name in {"td", "th"} and (node.get("colspan", "1") != "1" or node.get("rowspan", "1") != "1"):
            spans = {key: str(node.get(key, "1")) for key in ("colspan", "rowspan")}
            if any(not re.fullmatch(r"[1-9][0-9]{0,3}", value) for value in spans.values()):
                return GAP_MARKER
            return f'<{node.name} colspan="{spans["colspan"]}" rowspan="{spans["rowspan"]}">' + "".join(children) + f"</{node.name}>"
        if node.name == "tr":
            return "\t".join(part.strip("\n") for part in children) + "\n"
        text = "".join(children)
        if node.name in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "section", "article"}:
            return "\n" + text + "\n"
        return text

    text = render(body).replace("\r\n", "\n").replace("\r", "\n")
    if "\ufffd" in text:
        _fail("TEXT_DECODING", "HTML contains undecodable source characters")
    return text, title, links


def _pdf_graphics(content, resources, pdf, *, allow_forms: bool = True, depth: int = 0) -> bool:
    """Allow ordinary rules and text Form objects, including nested Forms."""
    if content is None:
        return False
    if depth > 8:
        return True
    resources = resources.get_object() if hasattr(resources, "get_object") else resources
    for operands, operator in ContentStream(content, pdf).operations:
        if operator in {b"sh", b"INLINE IMAGE", b"c", b"v", b"y"}:
            return True
        if operator == b"Do":
            if not allow_forms:
                # pypdf layout extraction can silently omit Form text.
                return True
            objects = resources.get("/XObject", {})
            objects = objects.get_object() if hasattr(objects, "get_object") else objects
            obj = objects.get(operands[0])
            obj = obj.get_object() if obj is not None else None
            if obj is None or obj.get("/Subtype") != "/Form":
                return True
            if _pdf_graphics(obj, obj.get("/Resources", resources), pdf, depth=depth + 1):
                return True
    return False


def _pdftotext_pages(raw: bytes, count: int) -> list[str]:
    """Run the explicitly configured local Poppler binary once per document."""
    with tempfile.TemporaryDirectory(prefix="taxcalcbench-pdf-") as directory:
        source, target = Path(directory) / "source.pdf", Path(directory) / "text.txt"
        source.write_bytes(raw)
        try:
            subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(source), str(target)],
                           check=True, timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise SourceError("PDF_BACKEND_MISSING", "Configured pdftotext backend requires Poppler on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SourceError("PDF_EXTRACTION_TIMEOUT", "pdftotext exceeded the 60-second document limit") from exc
        except subprocess.CalledProcessError as exc:
            raise SourceError("PDF_EXTRACTION_FAILED", "pdftotext could not extract the official PDF") from exc
        if not target.is_file():
            _fail("PDF_EXTRACTION_FAILED", "pdftotext did not produce a text file")
        if target.stat().st_size > 32 * 1024 * 1024:
            _fail("TEXT_LIMIT", "Extracted PDF text exceeds the byte limit")
        try:
            pages = target.read_text(encoding="utf-8").split("\f")
        except UnicodeError as exc:
            raise SourceError("TEXT_DECODING", "pdftotext output is not valid UTF-8") from exc
        if pages[-1] == "":
            pages.pop()
        if len(pages) != count:
            _fail("PDF_PAGE_MISMATCH", "pdftotext page boundaries do not match the source PDF")
        return pages


def _pdf(raw: bytes, max_pages: int, backend: str = "pypdf") -> tuple[str, list[dict]]:
    if not raw.startswith(b"%PDF-"):
        _fail("PDF_INVALID", "The response is not a PDF document")
    try:
        pdf = PdfReader(BytesIO(raw), strict=True)
        if pdf.is_encrypted and not pdf.decrypt(""):
            _fail("PDF_ENCRYPTED", "PDF requires a password")
        if not 0 < len(pdf.pages) <= max_pages:
            _fail("PDF_PAGE_LIMIT", "PDF exceeds the configured page limit")
        extracted_pages = _pdftotext_pages(raw, len(pdf.pages)) if backend == "pdftotext" else None
        output, boundaries, size, readable = [], [], 0, 0
        for number, page in enumerate(pdf.pages, 1):
            reason = None
            try:
                if _pdf_graphics(page.get_contents(), page.get("/Resources", {}), pdf,
                                 allow_forms=backend == "pdftotext"):
                    reason = "PDF_UNSUPPORTED_GRAPHICS"
                text = "" if reason else (extracted_pages[number - 1] if extracted_pages is not None
                    else (page.extract_text(extraction_mode="layout", layout_mode_strip_rotated=False) or ""))
                if not reason and (not text.strip() or "\ufffd" in text):
                    reason = "PDF_UNREADABLE_PAGE"
                if not reason and any(unicodedata.category(char) == "Cc" and char not in "\t\r\n" for char in text):
                    reason = "PDF_INVALID_FONT_MAPPING"
                if not reason and backend == "pypdf" and any(unicodedata.bidirectional(char) in {"R", "AL"} for char in text):
                    # pypdf can reverse Arabic digits and drop RTL phrases even
                    # when glyphs render correctly. Require the explicit RTL backend.
                    reason = "PDF_UNVERIFIED_RTL_ORDER"
            except Exception:
                text, reason = "", "PDF_UNREADABLE_PAGE"
            if reason:
                text = f"[UNAVAILABLE PDF PAGE {number}: {reason}; do not infer its provisions]"
            else:
                readable += 1
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            piece = f"[PDF PAGE {number}]\n{text}\n"
            boundary = {"kind": "page", "label": str(number), "start": size, "end": size + len(piece)}
            if reason:
                boundary["unavailable"] = reason
            boundaries.append(boundary)
            output.append(piece)
            size += len(piece)
            if size > 32 * 1024 * 1024:
                _fail("TEXT_LIMIT", "Extracted PDF text exceeds the character limit")
        if not readable:
            _fail("PDF_UNREADABLE", "PDF has no readable text pages without unsupported graphics")
        return "".join(output), boundaries
    except SourceError:
        raise
    except Exception as exc:
        raise SourceError("PDF_INVALID", "PDF could not be extracted reliably") from exc


def _extract(raw: bytes, mime: str, url: str, kind: str, policy: dict) -> dict:
    title, links, boundaries, warnings = "", [], [], []
    if mime in {"text/html", "application/xhtml+xml"}:
        selector = policy.get("html_selectors", {}).get(urlsplit(url).hostname) if kind != "navigation" else None
        text, title, links = _html(raw, selector,
            policy.get("html_remove_selectors", {}).get(urlsplit(url).hostname))
        if kind == "navigation":
            # Catalogue graphics are not offered as law evidence.
            text = text.replace(GAP_MARKER, "[CATALOGUE NON-TEXT CONTENT]")
    elif mime == DOCX_MIME:
        result = _extract_docx(raw, max_input_bytes=policy["max_bytes"])
        text, warnings = result.text, list(result.warnings)
    elif mime == "application/pdf":
        text, boundaries = _pdf(raw, policy["max_pdf_pages"], policy.get("pdf_text_backend", "pypdf"))
        warnings.append(f"PDF_TEXT_BACKEND: {policy.get('pdf_text_backend', 'pypdf')}")
        warnings.append("PDF_TEXT_LAYOUT; straight table/page rules are not transcribed")
    elif mime == "text/plain":
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise SourceError("TEXT_DECODING", "Plain source text must be UTF-8") from exc
    else:
        _fail("UNSUPPORTED_MEDIA_TYPE", "Official response is not supported HTML, PDF, DOCX or UTF-8 text")
    if not text.strip():
        _fail("NO_EXTRACTABLE_TEXT", "Official source contains no readable text")
    if len(text.encode("utf-8")) > 32 * 1024 * 1024:
        _fail("TEXT_LIMIT", "Extracted text exceeds the byte limit")
    text, gaps, sections = _withhold_gaps(text, policy.get("section_heading_pattern"))
    gaps.extend({"label": f"PDF page {part['label']}", "reason": part["unavailable"],
                 "start": part["start"], "end": part["end"]}
                for part in boundaries if part.get("unavailable"))
    if not boundaries:
        boundaries = sections
    elif sections:
        boundaries += sections
    return {"text": text, "title": title, "links": links, "boundaries": boundaries,
            "partial": bool(gaps), "extraction_gaps": gaps, "extraction_warnings": warnings}


_DEFAULTS = {"max_documents": 100, "max_pages": 150, "max_depth": 4, "max_links_per_page": 1000,
             "max_bytes": 20 * 1024 * 1024, "max_pdf_pages": 2000, "timeout_seconds": 20,
             "retries": 1, "max_redirects": 3}


def _policy(config: dict) -> dict:
    if not isinstance(config, dict) or not all(isinstance(config.get(k), str) and config[k].strip()
                                              for k in ("country", "language")):
        raise ValueError("Source configuration requires country and language")
    source = config.get("sources")
    if not isinstance(source, dict):
        raise ValueError("Source configuration requires a sources object")
    policy = {**_DEFAULTS, **source}
    if policy.get("pdf_text_backend", "pypdf") not in {"pypdf", "pdftotext"}:
        raise ValueError("PDF text backend must be pypdf or pdftotext")
    for key in _DEFAULTS:
        value = policy[key]
        if type(value) is not int or not (0 if key in {"max_depth", "retries", "max_redirects"} else 1) <= value:
            raise ValueError(f"Invalid source limit: {key}")
    if policy["retries"] > 5 or policy["max_redirects"] > 10 or policy["timeout_seconds"] > 300:
        raise ValueError("Retries, redirects or timeout exceed supported bounds")
    hosts = policy.get("allowed_hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError("An explicit official host/path allowlist is required")
    exclusions = policy.get("html_remove_selectors", {})
    if not isinstance(exclusions, dict) or any(host not in hosts or not isinstance(selectors, list)
        or any(not isinstance(selector, str) or not selector.strip() for selector in selectors)
        for host, selectors in exclusions.items()):
        raise ValueError("HTML exclusions require explicit selectors on an allowed official host")
    for host, paths in hosts.items():
        if (not isinstance(host, str) or host != host.lower() or not re.fullmatch(r"[a-z0-9.-]+", host)
                or not isinstance(paths, list) or not paths):
            raise ValueError("Each exact official hostname needs explicit allowed path patterns")
        for pattern in paths:
            re.compile(pattern)
    if policy.get("section_heading_pattern"):
        heading = re.compile(policy["section_heading_pattern"], re.MULTILINE)
        if heading.match(""):
            raise ValueError("Section heading pattern cannot match empty text")
    for rule in policy.get("link_rules", []):
        if rule.get("kind") not in {"law", "navigation", "guidance", "regulation", "treaty", "decision", "rate_table"}:
            raise ValueError("Each discovery rule needs an explicit document kind")
        re.compile(rule["url_pattern"])
        if type(rule.get("priority", 50)) is not int:
            raise ValueError("Link priority must be an integer")
    seeds = policy.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("At least one configured official seed is required")
    seen = set()
    for seed in seeds:
        _allowed(seed["url"], policy)
        if seed["url"] in seen or seed.get("kind") not in {
            "law", "navigation", "guidance", "regulation", "treaty", "decision", "rate_table"
        } or not isinstance(seed.get("title"), str) or not seed["title"].strip():
            raise ValueError("Seeds need unique URLs, titles and document kinds")
        seen.add(seed["url"])
        if seed.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", seed["sha256"]):
            raise ValueError("Seed checksum must be a lowercase SHA-256 digest")
    return policy


def _allowed(url: str, policy: dict) -> None:
    try:
        part = urlsplit(url)
        host, port = part.hostname, part.port
    except (TypeError, ValueError) as exc:
        raise SourceError("SOURCE_POLICY", "Source URL is malformed") from exc
    path = unquote(part.path)
    if (part.scheme != "https" or host not in policy["allowed_hosts"] or part.username or part.password
            or port not in (None, 443) or part.fragment or "\\" in url or any(ord(c) < 32 for c in url)
            or any(segment in {".", ".."} for segment in path.split("/"))
            or re.search(r"%(?:2f|5c|00|25)", part.path, re.IGNORECASE)
            or not any(re.fullmatch(pattern, part.path) for pattern in policy["allowed_hosts"].get(host, []))):
        _fail("SOURCE_POLICY", "Source URL is outside the configured exact official host/path policy")
    # Shared portals can bind a query as part of an exact URL rather than admit
    # arbitrary query-driven document routes under an otherwise approved path.
    if part.query and url not in policy.get("allowed_query_urls", []):
        _fail("SOURCE_POLICY", "Source query URL is not explicitly configured")


def _make_client(policy: dict) -> httpx.Client:
    """Use native trusted CAs; never disable certificate verification."""
    return httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), trust_env=False,
                        follow_redirects=False, timeout=policy["timeout_seconds"],
                        headers={"User-Agent": "TaxCalcBenchResearch/0.2", "Accept-Encoding": "identity"})


def _fetch(client: httpx.Client, url: str, policy: dict) -> tuple[bytes, str, dict, list[str]]:
    trace = []
    for _ in range(policy["max_redirects"] + 1):
        _allowed(url, policy)
        trace.append(url)
        for attempt in range(policy["retries"] + 1):
            try:
                with client.stream("GET", url, follow_redirects=False) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            _fail("REDIRECT_INVALID", "Official response has an empty redirect")
                        url = urljoin(url, location)
                        _allowed(url, policy)  # Check before making the next request.
                        break
                    if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < policy["retries"]:
                        time.sleep(min(0.25 * 2**attempt, 2))
                        continue
                    if response.status_code != 200:
                        _fail("HTTP_STATUS", f"Official source returned HTTP {response.status_code}")
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > policy["max_bytes"]:
                        _fail("DOWNLOAD_LIMIT", "Official source exceeds the download byte limit")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > policy["max_bytes"]:
                            _fail("DOWNLOAD_LIMIT", "Official source exceeds the download byte limit")
                        chunks.append(chunk)
                    return b"".join(chunks), url, dict(response.headers), trace
            except httpx.TimeoutException as exc:
                if attempt == policy["retries"]:
                    raise SourceError("CONNECT_TIMEOUT", "Official request timed out") from exc
                time.sleep(min(0.25 * 2**attempt, 2))
            except httpx.HTTPError as exc:
                if attempt == policy["retries"]:
                    raise SourceError("CONNECT_ERROR", "Official request failed with a transport or TLS error") from exc
                time.sleep(min(0.25 * 2**attempt, 2))
        else:
            _fail("CONNECT_ERROR", "Official request attempts were exhausted")
    _fail("REDIRECT_LIMIT", "Official request exceeded the redirect limit")


def _filename(url: str, headers: dict) -> str:
    value = None
    if headers.get("content-disposition"):
        message = Message()
        message["Content-Disposition"] = headers["content-disposition"]
        value = message.get_filename()
    value = value or unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "index.html"
    if not isinstance(value, str) or value in {".", ".."} or len(value) > 240 or any(
        char in value for char in ("/", "\\", "\x00")
    ) or any(ord(char) < 32 for char in value):
        _fail("UNSAFE_FILENAME", "Official filename is unsafe for local storage")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _config_sha(config: dict, policy: dict) -> str:
    payload = {"country": config["country"], "language": config["language"], "sources": policy}
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _save_inventory(source_dir: Path, inventory: dict) -> None:
    temporary = source_dir / "sources.json.tmp"
    temporary.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(source_dir / "sources.json")


def _local_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Source inventory paths must be relative")
    result = (root / value).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Source inventory path escapes its source folder")
    return result


def load_inventory(source_dir: Path, *, require_complete: bool = True) -> dict:
    """Load saved metadata and verify every retained original/text checksum."""
    root = Path(source_dir).resolve()
    try:
        inventory = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Source inventory is missing or invalid JSON") from exc
    if (not isinstance(inventory, dict) or inventory.get("version") != INVENTORY_VERSION
            or not isinstance(inventory.get("documents"), list) or not isinstance(inventory.get("failures"), list)
            or not isinstance(inventory.get("country"), str) or not isinstance(inventory.get("language"), str)):
        raise ValueError("Source inventory metadata is invalid")
    if require_complete and inventory.get("complete") is not True:
        raise ValueError("Source preparation was interrupted; rerun download to finish it")
    policy = _policy({"country": inventory["country"], "language": inventory["language"],
                      "sources": inventory.get("source_policy")})
    if _config_sha(inventory, policy) != inventory.get("sources_config_sha256"):
        raise ValueError("Source inventory policy checksum does not match")
    seen = set()
    for document in inventory["documents"]:
        if (not isinstance(document, dict) or not isinstance(document.get("id"), str)
                or document["id"] in seen or document.get("country") != inventory["country"]
                or document.get("language") != inventory["language"]
                or document.get("kind") not in {"law", "navigation", "guidance", "regulation", "treaty",
                                                "decision", "rate_table"}
                or not isinstance(document.get("title"), str) or not document["title"].strip()):
            raise ValueError("Source document metadata is invalid")
        seen.add(document["id"])
        _allowed(document.get("url"), policy)
        _allowed(document.get("final_url"), policy)
        if document["id"] != "s_" + _sha(document["url"].encode())[:20]:
            raise ValueError("Source document identifier does not match its URL")
        for path_key, hash_key in (("raw_path", "sha256"), ("text_path", "text_sha256")):
            try:
                data = _local_path(root, document.get(path_key)).read_bytes()
            except OSError as exc:
                raise ValueError("A source inventory file is missing") from exc
            if _sha(data) != document.get(hash_key):
                raise ValueError("A source inventory checksum does not match")
            if path_key == "text_path":
                try:
                    text = data.decode("utf-8", errors="strict")
                except UnicodeError as exc:
                    raise ValueError("Source text is not UTF-8") from exc
                if not text.strip() or GAP_MARKER in text:
                    raise ValueError("Source text contains unavailable unisolated material")
                for section in document.get("boundaries", []) + document.get("extraction_gaps", []):
                    if (type(section.get("start")) is not int or type(section.get("end")) is not int
                            or not 0 <= section["start"] < section["end"] <= len(text)):
                        raise ValueError("Source section offsets are invalid")
        if bool(document.get("extraction_gaps")) != document.get("partial", False):
            raise ValueError("Source partial-extraction metadata is inconsistent")
    for failure in inventory["failures"]:
        if not isinstance(failure, dict) or not isinstance(failure.get("code"), str):
            raise ValueError("Source failure metadata is invalid")
        if failure.get("raw_path"):
            if _sha(_local_path(root, failure["raw_path"]).read_bytes()) != failure.get("sha256"):
                raise ValueError("Retained unavailable source checksum does not match")
    return inventory


def policy_extension(previous: dict, current: dict) -> bool:
    """Accept additions without reinterpreting or removing saved evidence."""
    previous, current = {**_DEFAULTS, **previous}, {**_DEFAULTS, **current}
    for key in previous.keys() | current.keys():
        old, new = previous.get(key), current.get(key)
        if key in _DEFAULTS:
            if new < old:
                return False
        elif key == "allowed_hosts":
            if any(host not in new or not set(paths) <= set(new[host]) for host, paths in old.items()):
                return False
        elif key == "html_selectors":
            if any((old or {}).get(host) != (new or {}).get(host) for host in previous["allowed_hosts"]):
                return False
        elif key == "seeds":
            by_url = {seed["url"]: seed for seed in new}
            if any(by_url.get(seed["url"]) != seed for seed in old):
                return False
        elif key in {"link_rules", "allowed_query_urls"}:
            if any(item not in (new or []) for item in (old or [])):
                return False
        elif old != new:
            return False
    return True


def _prepare_inventory(config: dict, root: Path, policy: dict, limit: int) -> tuple[dict, bool]:
    signature = _config_sha(config, policy)
    if (root / "sources.json").exists():
        # Acquisition may resume an incomplete batch, after all retained files
        # and its original policy pass the same integrity checks as a full one.
        inventory = load_inventory(root, require_complete=False)
        if inventory["country"] != config["country"] or inventory["language"] != config["language"]:
            raise ValueError("Source policy changed country or language; choose a new source folder")
        changed = inventory["sources_config_sha256"] != signature or inventory["download_limit"] != limit
        if not changed:
            return inventory, False
        if limit < inventory["download_limit"] or not policy_extension(inventory["source_policy"], policy):
            raise ValueError("Source policy changed non-additively; choose a new source folder")
        by_url = {document["url"]: document for document in inventory["documents"]}
        for seed in policy["seeds"]:
            existing = by_url.get(seed["url"])
            if existing and (seed["kind"] != existing["kind"] or
                             seed.get("sha256", existing["sha256"]) != existing["sha256"]):
                raise ValueError("Source policy changed an acquired document definition; choose a new source folder")
        inventory.setdefault("policy_history", []).append({
            "source_policy": inventory["source_policy"], "sources_config_sha256": inventory["sources_config_sha256"],
            "download_limit": inventory["download_limit"]})
        inventory.update(source_policy=policy, sources_config_sha256=signature, download_limit=limit)
        return inventory, True
    if root.exists() and any(root.iterdir()):
        raise ValueError("Source folder has files without a valid inventory; choose a new folder")
    root.mkdir(parents=True, exist_ok=True)
    return {"version": INVENTORY_VERSION, "country": config["country"], "language": config["language"],
            "source_policy": policy, "sources_config_sha256": signature, "download_limit": limit,
            "created_at": datetime.now(timezone.utc).isoformat(), "complete": False,
            "prepared_policy_sha256": None,
            "documents": [], "failures": [], "pages_attempted": 0, "documents_attempted": 0}, True


def _observed_links(raw: bytes, record: dict, policy: dict) -> list[dict]:
    """Discover only allowlisted links matching a configured download rule."""
    if record.get("media_type") not in {"text/html", "application/xhtml+xml"}:
        return []
    final_url = record.get("final_url") or record["url"]
    candidates = {}
    for tag in BeautifulSoup(raw, "html.parser").find_all("a", href=True):
        href = tag.get("href")
        if not isinstance(href, str):
            continue
        try:
            parts = urlsplit(urljoin(final_url, href))
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
            _allowed(url, policy)
        except (ValueError, SourceError):
            continue
        if url in candidates:
            continue
        rule = next((rule for rule in sorted(policy.get("link_rules", []), key=lambda rule: rule.get("priority", 50))
                     if re.fullmatch(rule["url_pattern"], url)), None)
        if rule is None:
            continue
        label = tag.get_text(" ", strip=True)
        candidates[url] = {"url": url, "title": label or url, "kind": rule["kind"],
            "depth": record.get("depth", 0) + 1, "priority": rule.get("priority", 50),
            "observed_from": {"url": final_url, "sha256": record["sha256"], "raw_path": record["raw_path"]},
            "link_text": label}
    return sorted(candidates.values(), key=lambda item: (item["priority"], item["url"]))


def _candidate_records(inventory: dict, root: Path, policy: dict) -> list[dict]:
    acquired = {url for row in inventory["documents"] for url in (row["url"], row["final_url"])}
    failed = {row["url"] for row in inventory["failures"] if row.get("id") and row.get("url")}
    candidates = {seed["url"]: {**seed, "depth": 0, "priority": -1000 + index,
        "observed_from": seed.get("observed_from"), "link_text": None}
        for index, seed in enumerate(policy["seeds"])}
    records = [row for row in inventory["documents"] + inventory["failures"] if row.get("raw_path")]
    by_url = {url: row for row in records for url in (row["url"], row.get("final_url", row["url"]))}
    seed_urls = {seed["url"] for seed in policy["seeds"]}
    for record in sorted(records, key=lambda row: row["url"]):
        # Older inventories did not record crawl depth. Reconstruct it from
        # retained provenance only for discovered pages. Seed provenance may
        # be a descriptive string, not a discovery parent.
        if record["url"] in seed_urls or record.get("final_url") in seed_urls:
            depth = 0
        elif type(record.get("depth")) is int and record["depth"] >= 0:
            depth = record["depth"]
        else:
            depth, parent, visited = 0, record, set()
            while parent.get("observed_from") and parent["url"] not in seed_urls:
                provenance = parent["observed_from"]
                parent_url = provenance.get("url") if isinstance(provenance, dict) else provenance
                if not isinstance(parent_url, str) or parent_url in visited or parent_url not in by_url:
                    depth = policy["max_depth"]
                    break
                visited.add(parent_url)
                depth += 1
                parent = by_url[parent_url]
        if depth >= policy["max_depth"]:
            continue
        raw = _local_path(root, record["raw_path"]).read_bytes()
        if _sha(raw) != record["sha256"]:
            raise ValueError("Observed source checksum does not match")
        links = _observed_links(raw, record | {"depth": depth}, policy)
        for candidate in links[:policy["max_links_per_page"]]:
            candidates.setdefault(candidate["url"], candidate)
    for candidate in candidates.values():
        candidate["status"] = "acquired" if candidate["url"] in acquired else "unavailable" if candidate["url"] in failed else "available"
    return sorted(candidates.values(), key=lambda item: (item["priority"], item["depth"], item["url"]))


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            _fail("SOURCE_FILE_CONFLICT", "Acquisition would replace an existing source file")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete bytes; linking is atomic and never replaces a file
    # created by another process while this temporary file was being written.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.") as stream:
        stream.write(data)
        stream.flush()
        try:
            path.hardlink_to(stream.name)
        except FileExistsError:
            if path.read_bytes() != data:
                _fail("SOURCE_FILE_CONFLICT", "Acquisition would replace an existing source file")


def _run_acquisition(config: dict, root: Path, inventory: dict, queue: list[dict]) -> dict:
    policy, limit = inventory["source_policy"], inventory["download_limit"]
    seen = {row["url"] for row in inventory["documents"] + inventory["failures"] if row.get("url")}
    seen.update(item["url"] for item in queue)
    inventory["complete"] = False
    _save_inventory(root, inventory)
    with _make_client(policy) as client:
        while queue:
            queue.sort(key=lambda item: (item["priority"], item["depth"], item["url"]))
            item = queue.pop(0)
            if inventory["pages_attempted"] >= policy["max_pages"]:
                inventory["failures"].append({"code": "PAGE_LIMIT", "remaining_urls": len(queue) + 1})
                break
            if item["kind"] != "navigation" and inventory["documents_attempted"] >= limit:
                inventory["failures"].append({"code": "DOCUMENT_LIMIT", "remaining_urls": len(queue) + 1})
                break
            inventory["pages_attempted"] += 1
            inventory["documents_attempted"] += item["kind"] != "navigation"
            ident = "s_" + _sha(item["url"].encode())[:20]
            failure = {"url": item["url"], "observed_from": item["observed_from"], "id": ident}
            try:
                raw, final_url, headers, redirects = _fetch(client, item["url"], policy)
                seen.add(final_url)
                sha = _sha(raw)
                if item.get("sha256") and item["sha256"] != sha:
                    _fail("EXPECTED_CHECKSUM_MISMATCH", "Official source differs from its configured expected checksum")
                filename = _filename(final_url, headers)
                raw_path = Path("raw") / ident / filename
                _write_once(_local_path(root, raw_path.as_posix()), raw)
                mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
                failure.update(raw_path=raw_path.as_posix(), sha256=sha, final_url=final_url,
                               media_type=mime, depth=item["depth"], redirects=redirects)
                ordered = [row for row in _observed_links(raw, failure, policy) if row["url"] not in seen]
                if ordered and item["depth"] >= policy["max_depth"]:
                    inventory["failures"].append({"code": "DEPTH_LIMIT", "url": final_url, "remaining_urls": len(ordered)})
                else:
                    if len(ordered) > policy["max_links_per_page"]:
                        inventory["failures"].append({"code": "LINK_LIMIT", "url": final_url,
                                                       "remaining_urls": len(ordered) - policy["max_links_per_page"]})
                    for candidate in ordered[:policy["max_links_per_page"]]:
                        seen.add(candidate["url"])
                        queue.append(candidate)
                extracted = _extract(raw, mime, final_url, item["kind"], policy)
                text_path = Path("text") / f"{ident}.txt"
                text_bytes = extracted.pop("text").encode("utf-8")
                _write_once(_local_path(root, text_path.as_posix()), text_bytes)
                detected_title = extracted.pop("title")
                extracted.pop("links")
                document = {"id": ident, "title": item.get("title") or detected_title or item["url"],
                            "url": item["url"], "final_url": final_url, "redirects": redirects,
                            "raw_path": raw_path.as_posix(), "text_path": text_path.as_posix(),
                            "sha256": sha, "text_sha256": _sha(text_bytes), "media_type": mime,
                            "language": config["language"], "country": config["country"], "kind": item["kind"],
                            "downloaded_at": datetime.now(timezone.utc).isoformat(), "depth": item["depth"],
                            "observed_from": item["observed_from"], "link_text": item["link_text"],
                            "edition_note": item.get("edition_note", "Edition and applicability must be checked in source text"),
                            **extracted}
                inventory["documents"].append(document)
            except SourceError as exc:
                inventory["failures"].append({**failure, "code": exc.code, "message": str(exc)})
            _save_inventory(root, inventory)
    inventory["complete"] = True
    inventory["prepared_policy_sha256"] = inventory["sources_config_sha256"]
    inventory["bounded"] = any(row["code"].endswith("_LIMIT") for row in inventory["failures"])
    _save_inventory(root, inventory)
    return load_inventory(root)


def download_sources(config: dict, source_dir: Path, *, refresh: bool = False,
                     max_documents: int | None = None) -> dict:
    """Download official seeds/observed links, appending additive source updates.

    Existing raw/text files and IDs are retained. The unchanged completed cache
    makes no network requests. Replacing/removing policy or refreshing a source
    still requires a new folder. Limits count attempts across the collection.
    """
    policy = _policy(config)
    if max_documents is not None and (type(max_documents) is not int or max_documents < 1):
        raise ValueError("max_documents must be a positive integer")
    limit = max_documents if max_documents is not None else policy["max_documents"]
    root = Path(source_dir).resolve()
    if refresh and root.exists() and any(root.iterdir()):
        raise ValueError("Choose a new source folder for refresh")
    inventory, changed = _prepare_inventory(config, root, policy, limit)
    if (inventory.get("complete") is True and not changed
            and inventory.get("prepared_policy_sha256", inventory["sources_config_sha256"]) == inventory["sources_config_sha256"]):
        return inventory
    queue = [row for row in _candidate_records(inventory, root, policy) if row["status"] == "available"]
    return _run_acquisition(config, root, inventory, queue)

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

DATA_DIR = Path("data/eg/sources")
RAW_DIR = DATA_DIR / "raw"
TEXT_DIR = DATA_DIR / "text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/eg_ocr")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ocr_single_page(img_path: Path, page_num: int, max_retries: int = 5) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    for attempt in range(max_retries):
        try:
            res = await client.chat.completions.create(
                model="gemini-3.8-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe this official Egyptian Gazette legal page verbatim. "
                                "Preserve all Arabic text, article headings (المادة / مادة), dates, numbers, and layout. "
                                "Do not include conversational filler, preamble, or markdown codeblocks. Output only the transcribed text."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }]
            )
            content = res.choices[0].message.content or ""
            # Clean possible markdown wrap
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines)
            return content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = 2.0 * (attempt + 1)
            print(f"Page {page_num} attempt {attempt + 1} failed ({e}); retrying in {delay}s...")
            await asyncio.sleep(delay)
    return ""


async def process_law_91():
    doc_id = "s_331b14c4946110ba1ff0"
    raw_pdf = RAW_DIR / doc_id / "law_no.91-2005.pdf"
    if not raw_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {raw_pdf}")
    
    total_pages = 71
    first_img = TMP_DIR / "law91_page-01.png"
    if not first_img.exists():
        print(f"Rendering {total_pages} pages of Law 91/2005 via pdftoppm...")
        cmd = ["pdftoppm", "-png", "-r", "150", str(raw_pdf), str(TMP_DIR / "law91_page")]
        subprocess.run(cmd, check=True)
    else:
        print(f"Images already rendered in {TMP_DIR}")

    print(f"Transcribing {total_pages} pages with Gemini 3.8 Flash...")
    sem = asyncio.Semaphore(10)

    async def sem_ocr(p_num):
        cache_path = TMP_DIR / f"law91_page_{p_num}.txt"
        if cache_path.exists() and len(cache_path.read_text(encoding="utf-8").strip()) > 50:
            txt = cache_path.read_text(encoding="utf-8").strip()
            print(f"  Law 91/2005: Page {p_num}/{total_pages} cached ({len(txt)} chars)")
            return p_num, txt
        num_str = f"0{p_num}" if p_num < 10 else f"{p_num}"
        img_path = TMP_DIR / f"law91_page-{num_str}.png"
        async with sem:
            txt = await ocr_single_page(img_path, p_num)
            cache_path.write_text(txt, encoding="utf-8")
            print(f"  Law 91/2005: Page {p_num}/{total_pages} transcribed ({len(txt)} chars)")
            return p_num, txt

    tasks = [sem_ocr(i) for i in range(1, total_pages + 1)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])

    pieces = []
    boundaries = []
    current_offset = 0

    for p_num, page_text in results:
        piece = f"[PDF PAGE {p_num}]\n{page_text}\n\n"
        start = current_offset
        end = start + len(piece)
        boundaries.append({"kind": "page", "label": str(p_num), "start": start, "end": end})
        pieces.append(piece)
        current_offset = end

    full_text = "".join(pieces)
    text_file = TEXT_DIR / f"{doc_id}.txt"
    text_file.write_text(full_text, encoding="utf-8")
    print(f"Law 91/2005 written to {text_file} ({len(full_text)} chars, {len(boundaries)} pages)")

    return {
        "id": doc_id,
        "title": "قانون رقم 91 لسنة 2005 بإصدار قانون الضريبة على الدخل",
        "url": "https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.91-2005.pdf",
        "final_url": "https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.91-2005.pdf",
        "redirects": ["https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.91-2005.pdf"],
        "raw_path": f"raw/{doc_id}/law_no.91-2005.pdf",
        "text_path": f"text/{doc_id}.txt",
        "sha256": sha256_bytes(raw_pdf.read_bytes()),
        "text_sha256": sha256_text(full_text),
        "media_type": "application/pdf",
        "language": "ar",
        "country": "EG",
        "kind": "law",
        "downloaded_at": "2026-09-30T10:44:00+00:00",
        "depth": 0,
        "observed_from": "https://portal.eta.gov.eg/ar/content/qwanyn-aldrybt-ly-aldkhl",
        "link_text": "قانون رقم 91 لسنة 2005 بإصدار قانون الضريبة على الدخل",
        "edition_note": "نسخة رسمية منشورة من مصلحة الضرائب المصرية بالجريدة الرسمية العدد 23 (تابع) فى 9 يونية 2005.",
        "boundaries": boundaries,
        "partial": False,
        "extraction_gaps": [],
        "extraction_warnings": ["OCR_TRANSCRIPTION: poppler-pdftoppm-gemini-3.8-flash"]
    }


def process_law_26():
    doc_id = "s_dbeff943d02546ccd5aa"
    raw_pdf = RAW_DIR / doc_id / "law_no.26-2020.pdf"
    if not raw_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {raw_pdf}")

    out = subprocess.check_output(["pdftotext", "-layout", "-enc", "UTF-8", str(raw_pdf), "-"]).decode("utf-8")
    raw_pages = out.split("\f")
    if raw_pages and raw_pages[-1] == "":
        raw_pages.pop()

    pieces = []
    boundaries = []
    current_offset = 0

    for p_num, page_text in enumerate(raw_pages, 1):
        clean_text = page_text.strip()
        piece = f"[PDF PAGE {p_num}]\n{clean_text}\n\n"
        start = current_offset
        end = start + len(piece)
        boundaries.append({"kind": "page", "label": str(p_num), "start": start, "end": end})
        pieces.append(piece)
        current_offset = end

    full_text = "".join(pieces)
    text_file = TEXT_DIR / f"{doc_id}.txt"
    text_file.write_text(full_text, encoding="utf-8")
    print(f"Law 26/2020 written to {text_file} ({len(full_text)} chars, {len(boundaries)} pages)")

    return {
        "id": doc_id,
        "title": "قانون رقم 26 لسنة 2020 بتعديل بعض أحكام قانون الضريبة على الدخل",
        "url": "https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.26-2020.pdf",
        "final_url": "https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.26-2020.pdf",
        "redirects": ["https://portal.eta.gov.eg/sites/default/files/2024-03/law_no.26-2020.pdf"],
        "raw_path": f"raw/{doc_id}/law_no.26-2020.pdf",
        "text_path": f"text/{doc_id}.txt",
        "sha256": sha256_bytes(raw_pdf.read_bytes()),
        "text_sha256": sha256_text(full_text),
        "media_type": "application/pdf",
        "language": "ar",
        "country": "EG",
        "kind": "law",
        "downloaded_at": "2026-09-30T10:44:00+00:00",
        "depth": 0,
        "observed_from": "https://portal.eta.gov.eg/ar/content/qwanyn-aldrybt-ly-aldkhl",
        "link_text": "قانون رقم 26 لسنة 2020 بتعديل بعض أحكام قانون الضريبة على الدخل",
        "edition_note": "نسخة رسمية منشورة بالجريدة الرسمية العدد 19 (تابع) فى 7 مايو 2020.",
        "boundaries": boundaries,
        "partial": False,
        "extraction_gaps": [],
        "extraction_warnings": ["PDF_TEXT_BACKEND: pdftotext-with-header-image"]
    }


async def main():
    print("Processing Law 26 of 2020...")
    doc26 = process_law_26()

    print("Processing Law 91 of 2005 via Gemini OCR...")
    t0 = time.perf_counter()
    doc91 = await process_law_91()
    dt = time.perf_counter() - t0
    print(f"Law 91/2005 finished in {dt:.1f}s!")

    # Update sources.json
    sources_json_path = DATA_DIR / "sources.json"
    with open(sources_json_path) as f:
        inv = json.load(f)

    existing_docs = {d["id"]: d for d in inv.get("documents", [])}
    existing_docs[doc26["id"]] = doc26
    existing_docs[doc91["id"]] = doc91

    # Remove resolved failures
    resolved_ids = {doc26["id"], doc91["id"]}
    inv["failures"] = [f for f in inv.get("failures", []) if f["id"] not in resolved_ids]
    inv["documents"] = sorted(existing_docs.values(), key=lambda d: d["id"])

    with open(sources_json_path, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully updated {sources_json_path}!")
    print(f"Total active documents: {len(inv['documents'])}")
    for d in inv["documents"]:
        print(f" - {d['id']}: {d['title']} ({len(d['boundaries'])} pages)")


if __name__ == "__main__":
    asyncio.run(main())

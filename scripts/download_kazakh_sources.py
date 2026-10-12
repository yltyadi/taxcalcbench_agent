"""Download and process official Kazakh statutory texts corresponding to Kazakhstan citations."""

import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from taxcalcbench.sources import _html

SOURCE_DIR = Path("data/kz_kk/sources")
RAW_DIR = SOURCE_DIR / "raw"
TEXT_DIR = SOURCE_DIR / "text"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def download_one(item: dict, timeout: int = 60, max_retries: int = 3):
    url = item["kk_url"]
    doc_id = item["id"]
    filename = item["filename"]
    save_raw = RAW_DIR / doc_id / filename
    save_text = TEXT_DIR / f"{doc_id}.txt"

    if save_raw.exists() and save_text.exists():
        print(f"[CACHE] {doc_id} already exists ({save_text.stat().st_size} bytes)")
        raw = save_raw.read_bytes()
        text = save_text.read_text(encoding="utf-8")
        return {
            **item,
            "raw_path": str(save_raw.relative_to(SOURCE_DIR)),
            "text_path": str(save_text.relative_to(SOURCE_DIR)),
            "sha256": sha256_bytes(raw),
            "text_sha256": sha256_text(text),
            "size": len(raw),
            "status": "cached",
        }

    save_raw.parent.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[FETCH] Starting {doc_id}: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "kk,ru,en;q=0.9",
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            elapsed = time.time() - t0
            save_raw.write_bytes(raw)

            # Extract clean text using taxcalcbench _html
            text, title, _ = _html(raw, selector="article", remove_selectors=None)
            save_text.write_text(text, encoding="utf-8")

            print(f"[DONE] {doc_id} in {elapsed:.1f}s ({len(raw)} bytes raw, {len(text)} chars text): {title[:50]}")
            return {
                **item,
                "title": title or item["ru_title"],
                "raw_path": str(save_raw.relative_to(SOURCE_DIR)),
                "text_path": str(save_text.relative_to(SOURCE_DIR)),
                "sha256": sha256_bytes(raw),
                "text_sha256": sha256_text(text),
                "size": len(raw),
                "status": "downloaded",
            }
        except Exception as exc:
            last_err = exc
            print(f"[RETRY {attempt}/{max_retries}] {doc_id} failed: {exc}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download {doc_id} from {url}: {last_err}")


def main():
    with open("outputs/full/kz/citations.json") as f:
        cits = json.load(f)
    with open("data/kz/sources/sources.json") as f:
        ru_sources = json.load(f)

    url_to_source = {d["url"]: d for d in ru_sources["documents"]}
    used_sources = {}
    for c in cits:
        u = c["official_source_url"]
        if u in url_to_source:
            doc = url_to_source[u]
            doc_id = doc["id"]
            if doc_id not in used_sources:
                filename = Path(doc["raw_path"]).name
                used_sources[doc_id] = {
                    "id": doc_id,
                    "ru_url": u,
                    "kk_url": u.replace("/rus/", "/kaz/"),
                    "ru_title": doc["title"],
                    "filename": filename,
                    "country": "KZ",
                    "language": "kk",
                    "citation_count": 0,
                }
            used_sources[doc_id]["citation_count"] += 1

    print(f"Found {len(used_sources)} unique statutory sources cited across {len(cits)} citations.")

    items = list(used_sources.values())
    results = []

    # Use 3 concurrent workers to avoid overloading adilet
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_one, item): item for item in items}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)

    results.sort(key=lambda x: x["id"])

    # Save sources.json metadata
    sources_meta = {
        "version": "1.0",
        "country": "KZ",
        "language": "kk",
        "documents": results,
    }
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_DIR / "sources.json", "w", encoding="utf-8") as f:
        json.dump(sources_meta, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully prepared data/kz_kk/sources/ with {len(results)} official Kazakh statutory texts!")


if __name__ == "__main__":
    main()

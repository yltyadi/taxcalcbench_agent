"""Additive official-source acquisition, using synthetic HTTP fixtures only."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from taxcalcbench import sources

BASE = "https://official.example/national/"


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    config = {"country": "SYNTHETIC", "language": "xx", "sources": {
        "allowed_hosts": {"official.example": [r"/national/[a-z0-9._/-]+"]},
        "seeds": [{"url": BASE + "start.html", "title": "Synthetic index", "kind": "navigation"}],
        "link_rules": [{"url_pattern": r"https://official\.example/national/code\.html", "kind": "law"}],
        "section_heading_pattern": r"^Article [0-9]+\.", "retries": 0,
        "max_documents": 10, "max_pages": 20, "max_depth": 4}}
    calls, responses = [], {
        BASE + "start.html": b'<p>Synthetic official catalogue <a href="code.html">Code</a></p>'
                             b'<p>Annual synthetic thresholds <a href="budget.html">Budget</a></p>',
        BASE + "code.html": b'<article><p>Article 1. SYNTHETIC CODE</p><p>Not actual law.</p></article>',
        BASE + "budget.html": b'<article><p>Article 2. SYNTHETIC BUDGET</p><p>Not actual law.</p></article>',
    }

    def handle(request):
        url = str(request.url)
        calls.append(url)
        response = responses[url]
        return httpx.Response(200, content=response, headers={"content-type": "text/html"}) if isinstance(response, bytes) else response

    monkeypatch.setattr(sources, "_make_client", lambda policy: httpx.Client(transport=httpx.MockTransport(handle)))
    inventory = sources.download_sources(config, tmp_path)
    assert calls == [BASE + "start.html", BASE + "code.html"]
    return config, tmp_path, inventory, responses, calls


def saved_files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for folder in (root / "raw", root / "text") for path in folder.rglob("*") if path.is_file()}


def test_additive_seeds_and_observed_children_append_in_same_folder(corpus):
    config, root, before, responses, calls = corpus
    originals = saved_files(root)
    updated = deepcopy(config)
    updated["sources"]["seeds"].append({"url": BASE + "extra.html", "title": "Extra synthetic", "kind": "law"})
    updated["sources"]["link_rules"].append({"url_pattern": r"https://official\.example/national/child\.html", "kind": "regulation"})
    responses[BASE + "extra.html"] = b'<article><p>Article 3. Synthetic</p><a href="child.html">Related instrument</a></article>'
    responses[BASE + "child.html"] = b'<article><p>Article 4. Synthetic regulation</p></article>'
    inventory = sources.download_sources(updated, root)
    assert calls == [BASE + "start.html", BASE + "code.html", BASE + "extra.html", BASE + "child.html"]
    assert inventory["documents"][:2] == before["documents"]
    assert all(saved_files(root)[path] == data for path, data in originals.items())
    child = inventory["documents"][-1]
    assert child["observed_from"]["url"] == BASE + "extra.html" and child["kind"] == "regulation"
    assert inventory["pages_attempted"] == 4 and inventory["documents_attempted"] == 3
    assert inventory["source_policy"] == sources._policy(updated)
    assert inventory["policy_history"][0]["source_policy"] == before["source_policy"]
    assert sources.download_sources(updated, root) == inventory and len(calls) == 4


def test_additive_host_path_and_selector_for_new_host_are_supported(corpus):
    config, root, before, responses, calls = corpus
    updated = deepcopy(config)
    updated["sources"]["allowed_hosts"]["second.example"] = [r"/national/law\.html"]
    updated["sources"]["html_selectors"] = {"second.example": "article"}
    url = "https://second.example/national/law.html"
    updated["sources"]["seeds"].append({"url": url, "title": "Second synthetic source", "kind": "law"})
    responses[url] = b'<nav>Navigation only</nav><article><p>Article 5. Synthetic second source</p></article>'
    inventory = sources.download_sources(updated, root)
    assert inventory["documents"][:2] == before["documents"] and calls[-1] == url
    assert "Navigation only" not in (root / inventory["documents"][-1]["text_path"]).read_text()


@pytest.mark.parametrize("change", ["remove_seed", "replace_seed", "remove_path", "heading", "selector", "limit", "rule", "country", "language"])
def test_non_additive_updates_fail_before_mutating_or_fetching(corpus, change):
    config, root, _, _, calls = corpus
    updated = deepcopy(config)
    if change == "remove_seed":
        updated["sources"]["seeds"] = [{"url": BASE + "budget.html", "title": "Replacement", "kind": "law"}]
    elif change == "replace_seed":
        updated["sources"]["seeds"][0]["title"] = "Changed definition"
    elif change == "remove_path":
        updated["sources"]["allowed_hosts"]["official.example"] = [r"/national/start\.html"]
    elif change == "heading":
        updated["sources"]["section_heading_pattern"] = r"^Different heading"
    elif change == "selector":
        updated["sources"]["html_selectors"] = {"official.example": "article"}
    elif change == "limit":
        updated["sources"]["max_documents"] = 9
    elif change == "rule":
        updated["sources"]["link_rules"] = []
    else:
        updated[change] = "different"
    old_files, old_inventory, old_calls = saved_files(root), (root / "sources.json").read_bytes(), list(calls)
    with pytest.raises(ValueError):
        sources.download_sources(updated, root)
    assert calls == old_calls and saved_files(root) == old_files and (root / "sources.json").read_bytes() == old_inventory


def test_policy_extension_accepts_implicit_defaults_but_no_removed_query_permissions(corpus):
    config = corpus[0]
    original = sources._policy(config)
    assert sources.policy_extension(config["sources"], original)
    updated = deepcopy(original)
    updated["allowed_query_urls"] = [BASE + "budget.html?edition=one"]
    assert sources.policy_extension(original, updated)
    assert not sources.policy_extension(updated, original)


def test_higher_document_limit_continues_pending_seeds_without_rewriting(corpus):
    config, root, before, _, calls = corpus
    lower_root = root / "separate-smoke"
    first = sources.download_sources(config, lower_root, max_documents=1)
    old_files = saved_files(lower_root)
    updated = deepcopy(config)
    updated["sources"]["seeds"].append({"url": BASE + "budget.html", "title": "Budget", "kind": "law"})
    result = sources.download_sources(updated, lower_root)
    assert result["documents"][:2] == first["documents"] and result["download_limit"] == 10
    assert result["documents"][-1]["url"] == BASE + "budget.html"
    assert all(saved_files(lower_root)[path] == data for path, data in old_files.items())
    assert len(before["documents"]) == 2 and len(calls) == 5



def test_download_requires_configured_rule_before_following_an_observed_link(corpus):
    config, root, before, _, calls = corpus
    assert sources.download_sources(config, root) == before
    assert calls == [BASE + "start.html", BASE + "code.html"]
    updated = deepcopy(config)
    updated["sources"]["link_rules"].append({
        "url_pattern": r"https://official\.example/national/budget\.html", "kind": "law"})
    result = sources.download_sources(updated, root)
    assert calls == [BASE + "start.html", BASE + "code.html", BASE + "budget.html"]
    assert result["documents"][:2] == before["documents"]
    assert result["documents"][-1]["observed_from"]["url"] == BASE + "start.html"


def test_additive_download_verifies_originals_before_network(corpus):
    config, root, inventory, _, calls = corpus
    (root / inventory["documents"][0]["raw_path"]).write_bytes(b"TAMPERED SYNTHETIC SOURCE")
    updated = deepcopy(config)
    updated["sources"]["seeds"].append({"url": BASE + "budget.html", "title": "Budget", "kind": "law"})
    with pytest.raises(ValueError, match="checksum"):
        sources.download_sources(updated, root)
    assert len(calls) == 2


def test_additive_download_never_overwrites_an_untracked_original(corpus):
    config, root, before, _, calls = corpus
    ident = "s_" + sha256((BASE + "budget.html").encode()).hexdigest()[:20]
    original = root / "raw" / ident / "budget.html"
    original.parent.mkdir()
    original.write_bytes(b"EARLIER RETAINED SYNTHETIC BYTES")
    updated = deepcopy(config)
    updated["sources"]["seeds"].append({"url": BASE + "budget.html", "title": "Budget", "kind": "law"})
    result = sources.download_sources(updated, root)
    assert original.read_bytes() == b"EARLIER RETAINED SYNTHETIC BYTES"
    assert result["documents"] == before["documents"] and len(calls) == 3
    assert result["failures"][-1]["code"] == "SOURCE_FILE_CONFLICT"


@pytest.mark.parametrize("provenance", ["Configured source note", {"url": "https://official.example/unretained-index"}])
def test_configured_seed_keeps_depth_zero_when_download_rules_expand(corpus, provenance):
    config, root, _, _, calls = corpus
    inventory = sources.load_inventory(root)
    seed = inventory["documents"][0]
    seed["observed_from"] = provenance
    seed["depth"] = inventory["source_policy"]["max_depth"]
    sources._save_inventory(root, inventory)
    updated = deepcopy(config)
    updated["sources"]["link_rules"].append({
        "url_pattern": r"https://official\.example/national/budget\.html", "kind": "law"})
    result = sources.download_sources(updated, root)
    assert result["documents"][-1]["depth"] == 1
    assert result["documents"][-1]["observed_from"]["url"] == seed["url"]
    assert calls[-1] == BASE + "budget.html"


@pytest.mark.parametrize("additive", [False, True])
def test_interrupted_download_resumes_only_remaining_documents(corpus, monkeypatch, additive):
    config, root, before, responses, calls = corpus
    config = deepcopy(config)
    if additive:
        config["sources"]["seeds"] += [
            {"url": BASE + name, "title": "Synthetic added law", "kind": "law"}
            for name in ("budget.html", "extra.html")]
        responses[BASE + "extra.html"] = b'<p>Article 3. SYNTHETIC EXTRA LAW</p>'
        stop_count, remaining = 3, BASE + "extra.html"
    else:
        root = root / "interrupted-new-collection"
        stop_count, remaining = 1, BASE + "code.html"
    save = sources._save_inventory

    def interrupt_after_document(path, inventory):
        save(path, inventory)
        if len(inventory["documents"]) == stop_count:
            raise KeyboardInterrupt("SYNTHETIC interrupted acquisition")

    monkeypatch.setattr(sources, "_save_inventory", interrupt_after_document)
    with pytest.raises(KeyboardInterrupt):
        sources.download_sources(config, root)
    partial = json.loads((root / "sources.json").read_text())
    assert partial["complete"] is False
    with pytest.raises(ValueError, match="interrupted"):
        sources.load_inventory(root)
    retained, old_calls = saved_files(root), list(calls)
    # An old preparation marker must not make an incomplete batch look ready.
    partial["prepared_policy_sha256"] = partial["sources_config_sha256"]
    save(root, partial)
    monkeypatch.setattr(sources, "_save_inventory", save)
    result = sources.download_sources(config, root)
    assert calls == old_calls + [remaining]
    assert result["complete"] is True
    assert result["documents"][:stop_count] == partial["documents"]
    assert all(saved_files(root)[path] == data for path, data in retained.items())
    assert sources.load_inventory(root) == result


@pytest.mark.parametrize("field", ["raw_path", "text_path"])
def test_interrupted_download_checks_retained_hashes_before_fetch(corpus, field):
    config, root, inventory, _, calls = corpus
    inventory["complete"] = False
    sources._save_inventory(root, inventory)
    (root / inventory["documents"][0][field]).write_bytes(b"TAMPERED SYNTHETIC SOURCE")
    with pytest.raises(ValueError, match="checksum"):
        sources.download_sources(config, root)
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["country", "seed", "heading"])
def test_interrupted_download_keeps_policy_guards(corpus, change):
    config, root, inventory, _, calls = corpus
    inventory["complete"] = False
    sources._save_inventory(root, inventory)
    config = deepcopy(config)
    if change == "country":
        config["country"] = "DIFFERENT"
    elif change == "seed":
        config["sources"]["seeds"][0]["title"] = "Changed definition"
    else:
        config["sources"]["section_heading_pattern"] = r"^Changed heading"
    before = (root / "sources.json").read_bytes()
    with pytest.raises(ValueError):
        sources.download_sources(config, root)
    assert len(calls) == 2 and (root / "sources.json").read_bytes() == before


def test_interrupted_source_write_never_publishes_partial_bytes(tmp_path, monkeypatch):
    target = tmp_path / "raw" / "law.html"

    def interrupt_publication(path, temporary):
        assert Path(temporary).read_bytes() == b"COMPLETE SYNTHETIC DOCUMENT"
        assert not path.exists()
        raise KeyboardInterrupt("SYNTHETIC stop before atomic publication")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "hardlink_to", interrupt_publication)
        with pytest.raises(KeyboardInterrupt):
            sources._write_once(target, b"COMPLETE SYNTHETIC DOCUMENT")
    assert not target.exists() and not list(target.parent.iterdir())
    sources._write_once(target, b"COMPLETE SYNTHETIC DOCUMENT")
    assert target.read_bytes() == b"COMPLETE SYNTHETIC DOCUMENT"


def test_atomic_publication_cannot_replace_a_racing_source_file(tmp_path, monkeypatch):
    target = tmp_path / "law.html"
    link = Path.hardlink_to

    def publish_other_file(path, temporary):
        path.write_bytes(b"EXISTING SYNTHETIC EVIDENCE")
        link(path, temporary)

    monkeypatch.setattr(Path, "hardlink_to", publish_other_file)
    with pytest.raises(sources.SourceError, match="replace an existing"):
        sources._write_once(target, b"NEW SYNTHETIC EVIDENCE")
    assert target.read_bytes() == b"EXISTING SYNTHETIC EVIDENCE"

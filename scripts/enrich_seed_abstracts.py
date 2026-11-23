#!/usr/bin/env python3
"""Backfill missing seed/related-paper abstracts.

OpenAlex lacks abstracts for a handful of paywalled papers (6 of the faculty
paper's 18 seed references + 2 related works). This script patches them from
Crossref first (reliable, polite pool), then Semantic Scholar (rate-limited).
Polite pacing: 1 req / 2 s, exponential backoff on 429.
"""
import difflib
import glob
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "literature_corpus")

S2_DOI = "https://api.semanticscholar.org/graph/v1/paper/DOI:{}?fields=title,abstract"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search?fields=title,abstract&limit=3&query={}"
CR_DOI = "https://api.crossref.org/works/{}"
CR_SEARCH = "https://api.crossref.org/works?rows=3&query.bibliographic={}"
CR_HEADERS = {"User-Agent": "literature-corpus-builder/1.0 (mailto:research@example.org)"}


def parse_md(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return None, None, None
    end = text.find("\n---", 3)
    if end == -1:
        return None, None, None
    fm = json.loads(text[3:end])
    rest = text[end + 4:]
    m = re.search(r"## Abstract\s*\n(.*?)(?=\n## |\Z)", rest, re.S)
    abstract = m.group(1).strip() if m else ""
    return fm, abstract, rest


def strip_jats(x):
    return re.sub(r"<[^>]+>", "", x or "").strip()


def crossref_abstract(fm):
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", fm.get("doi") or "")
    try:
        if doi:
            r = requests.get(CR_DOI.format(requests.utils.quote(doi, safe="")),
                             timeout=30, headers=CR_HEADERS)
            if r.status_code == 200:
                a = strip_jats(((r.json().get("message") or {}).get("abstract")) or "")
                if a:
                    return a, "crossref"
        if fm.get("title"):
            r = requests.get(CR_SEARCH.format(requests.utils.quote(fm["title"])),
                             timeout=30, headers=CR_HEADERS)
            if r.status_code == 200:
                items = ((r.json().get("message") or {}).get("items")) or []
                best, score = None, 0.0
                for it in items:
                    t = (it.get("title") or [""])[0]
                    s = difflib.SequenceMatcher(
                        None, (fm["title"] or "").lower(), (t or "").lower()).ratio()
                    if s > score:
                        best, score = it, s
                if best and score > 0.85:
                    a = strip_jats(best.get("abstract") or "")
                    if a:
                        return a, "crossref"
    except requests.RequestException:
        pass
    return None, None


def s2_get(url, retries=4):
    for i in range(retries):
        try:
            r = requests.get(
                url, timeout=30,
                headers={"Accept-Encoding": "gzip, deflate",
                         "User-Agent": "literature-corpus-builder/1.0 (research)"})
            if r.status_code == 429:
                time.sleep(4 + 2 * i)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** i)
    return None


def s2_abstract(fm):
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", fm.get("doi") or "")
    if doi:
        d = s2_get(S2_DOI.format(requests.utils.quote(doi, safe="")))
        if d and d.get("abstract"):
            return d["abstract"], "semantic scholar"
    if fm.get("title"):
        d = s2_get(S2_SEARCH.format(requests.utils.quote(fm["title"])))
        papers = (d or {}).get("data") or []
        best, score = None, 0.0
        for p in papers:
            s = difflib.SequenceMatcher(
                None, (fm["title"] or "").lower(), (p.get("title") or "").lower()).ratio()
            if s > score:
                best, score = p, s
        if best and score > 0.75 and best.get("abstract"):
            return best["abstract"], "semantic scholar"
    return None, None


def abstract_for(fm):
    for fn in (crossref_abstract, s2_abstract):
        text, source = fn(fm)
        if text:
            return text, source
    return None, None


def main():
    patched, failed = [], []
    for path in sorted(glob.glob(os.path.join(CORPUS, "*.md"))):
        fm, abstract, rest = parse_md(path)
        if not fm:
            continue
        src = str(fm.get("source", ""))
        if not (fm.get("seed") or src.startswith("related:")):
            continue
        if abstract and "No abstract available" not in abstract:
            continue
        abs2, asrc = abstract_for(fm)
        if not abs2:
            failed.append(fm.get("title"))
            time.sleep(1.0)
            continue
        abs2 = " ".join(abs2.split())
        new_rest, n = re.subn(
            r"(## Abstract\s*\n).*?(\n\n## )",
            lambda m: m.group(1) + abs2 + m.group(2),
            rest,
            flags=re.S,
        )
        if n != 1:
            failed.append((fm.get("title"), "section-regex"))
            continue
        new_fm = dict(fm)
        new_fm["abstract_words"] = len(abs2.split())
        new_fm["abstract_source"] = asrc
        new_text = (
            "---\n"
            + json.dumps(new_fm, indent=2, ensure_ascii=False)
            + "\n---\n"
            + new_rest
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        patched.append(fm.get("title"))
        time.sleep(1.0)

    print(json.dumps({"patched": patched, "failed": failed}, indent=2))


if __name__ == "__main__":
    main()

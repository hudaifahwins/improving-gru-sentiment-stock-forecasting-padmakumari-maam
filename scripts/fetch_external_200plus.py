#!/usr/bin/env python3
"""Fetch 200+ real scholarly paper records via public APIs (OpenAlex primary).

Themes: GRU/LSTM stock forecasting, return vs. price-level targets, multi-day
horizons, evaluation leakage, financial news/GDELT sentiment.

Integrity rules:
  - Only exact fields returned by the API are stored; no fabrication.
  - Abstracts are reconstructed from OpenAlex abstract_inverted_index verbatim.
  - OA status / OA URL / abstract availability are recorded per record.
  - Dedup by OpenAlex ID AND normalized title; core-finance token guard applied.

Outputs (gitignored data/ folder):
  - data/external_200plus_papers.jsonl
  - data/external_200plus_papers_summary.md

Usage: python3 scripts/fetch_external_200plus.py --target 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import requests

API = "https://api.openalex.org"

QUERIES = [
    "stock market prediction deep learning",
    "stock return prediction neural network",
    "LSTM stock market forecasting",
    "GRU stock forecasting",
    "financial sentiment analysis time series",
    "news sentiment stock market forecasting",
    "stock forecasting multi-horizon",
    "financial time series deep learning evaluation",
    "trading signal deep learning leakage",
    "stock price prediction return level",
    "sentiment time series financial markets",
    "financial news forecasting deep learning",
]

CORE_RE = re.compile(
    r"\bstocks?\b|equit\w*|financ\w*|finans\w*|invest(or|ment)s?|investing|portfolio|crypt\w*|bitcoin|"
    r"forex|\bassets?\b|\btrading\b|\bbonds?\b|share price|closing price|index price|"
    r"market index|stock index|exchange rate|stock exchange|nasdaq|nepse|bist|masi|lq45|fintech",
    re.I,
)


def get(url, params=None, retries=4):
    headers = {"Accept-Encoding": "gzip, deflate"}
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params = dict(params or {})
        params["mailto"] = mailto
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**i + 1)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2**i)
    raise RuntimeError("unreachable")


def reconstruct_abstract(inv):
    if not inv:
        return ""
    words = []
    for word, positions in inv.items():
        for p in positions:
            if p >= len(words):
                words.extend([""] * (p - len(words) + 1))
            words[p] = word
    return " ".join(words)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    works = {}
    provenance = {}
    stats = {"dropped_dupe_id": 0, "dropped_dupe_title": 0, "dropped_offdomain": 0}
    dropped = []

    def add_work(w, source):
        wid = w["id"].rsplit("/", 1)[-1]
        if wid in works:
            stats["dropped_dupe_id"] += 1
            return False
        nt = norm_title(w.get("title"))
        if nt and nt != "untitled" and nt in {norm_title(x.get("title")) for x in works.values()}:
            stats["dropped_dupe_title"] += 1
            return False
        if not CORE_RE.search(" ".join([w.get("title") or "", reconstruct_abstract(w.get("abstract_inverted_index")),
                                       " ".join(t.get("display_name", "") for t in (w.get("topics") or []))])):
            stats["dropped_offdomain"] += 1
            dropped.append({"openalex_id": wid, "title": w.get("title"), "source": source, "reason": "offdomain"})
            return False
        works[wid] = w
        provenance[wid] = source
        return True

    base_filter = "has_abstract:true,is_retracted:false,from_publication_date:2018-01-01"
    for q in QUERIES:
        if len(works) >= args.target:
            break
        params = {"filter": f"title_and_abstract.search:{q},{base_filter}", "per-page": 200}
        data = get(f"{API}/works", params)
        for w in data.get("results", []):
            if len(works) >= args.target:
                break
            add_work(w, f"search:{q}")
        print(f"[search '{q}'] -> added {len(works)} distinct so far")
        time.sleep(0.15)

    entries = []
    for wid, w in sorted(works.items(), key=lambda kv: (kv[1].get("publication_year") or 0, kv[1].get("title") or "")):
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        oa = w.get("open_access") or {}
        entries.append({
            "openalex_id": wid,
            "doi": w.get("doi") or "",
            "title": w.get("title") or "Untitled",
            "authors": [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])],
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
            "abstract": abstract,
            "abstract_available": bool(abstract.strip()),
            "oa_url": oa.get("oa_url") or "",
            "is_oa": bool(oa.get("is_oa")),
            "cited_by_count": w.get("cited_by_count") or 0,
            "topics": [t.get("display_name", "") for t in (w.get("topics") or [])[:6]],
            "source_query": provenance.get(wid, "unknown"),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    jpath = os.path.join(args.out_dir, "external_200plus_papers.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n = len(entries)
    with_abs = sum(1 for e in entries if e["abstract_available"])
    oa_n = sum(1 for e in entries if e["is_oa"])
    years = sorted({e["year"] for e in entries if e["year"]})
    summary = f"""# External 200+ Papers — OpenAlex fetch (GRU/LSTM stock forecasting, return vs price-level targets, multi-day horizons, leakage, GDELT sentiment)

- Source: OpenAlex (`https://api.openalex.org`) via `scripts/fetch_external_200plus.py`
- Queries used: {len(QUERIES)} (`{' ; '.join(QUERIES)}`)
- Distinct records fetched/saved: **{n}**
- With abstract: **{with_abs}** ({with_abs / max(n,1):.1%}) · Open access: **{oa_n}** ({oa_n / max(n,1):.1%})
- Year range: {years[0]}–{years[-1] if years else 'n/a'}
- Dropped (dedup/off-domain): {stats}

Note: abstracts come from OpenAlex `abstract_inverted_index` (verbatim); full-text reads were not performed (mostly paywalled). No bibliographic fields were invented.
"""
    spath = os.path.join(args.out_dir, "external_200plus_papers_summary.md")
    with open(spath, "w", encoding="utf-8") as f:
        f.write(summary)

    print(json.dumps({
        "saved": n,
        "with_abstract": with_abs,
        "open_access": oa_n,
        "dropped_offdomain": stats["dropped_offdomain"],
        "dropped_dupe_id": stats["dropped_dupe_id"],
        "dropped_dupe_title": stats["dropped_dupe_title"],
        "year_range": [years[0], years[-1]] if years else None,
        "jsonl": jpath,
        "summary_md": spath,
    }, indent=2))


if __name__ == "__main__":
    main()

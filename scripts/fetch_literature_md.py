#!/usr/bin/env python3
"""Fetch a literature corpus from OpenAlex into per-paper Markdown files. (v4)

Corpus composition — relevance-first, tied to the faculty paper
(Padmakumari et al. 2026, "Precision Investing", OpenAlex W7171375780):
  1. Seed: the faculty paper's 18 OpenAlex references (its own bibliography).
  2. Lineage: works citing those seeds (filter=cites:), relevance-ranked,
     passed through the core finance-token guard.
  3. Faculty paper's own OpenAlex related_works (10; guard-exempt by
     definition — OpenAlex computed them as similar works).
  4. Topic backfill: title_and_abstract.search (title/abstract-only matching —
     tight relevance) over domain queries until the target count is reached.

Guards: dedup by OpenAlex ID AND normalized title; non-exempt papers must
mention a CORE finance token in title+abstract+topics (stock/equity/financial/
investment/portfolio/crypto/bitcoin/forex/asset/trading/bond/share price/
closing price/index price/market index/exchange rate/stock exchange or the
NEPSE/NASDAQ/BIST/MASI/LQ45 market names). This kills lineage false positives
(horticulture DL reviews, water-quality LSTM, hate-speech BiLSTM...) while
keeping genuine finance papers including non-English titles.

OpenAlex without an API key allows only page 1 (200 results) per query, so
search queries are harvested under two sort orders (relevance, most-cited).

Every .md has YAML frontmatter (JSON) + abstract + links. Full text is not
bulk-downloaded (mostly paywalled); OA links are recorded. Seed/related
abstracts missing from OpenAlex can be backfilled via
scripts/enrich_seed_abstracts.py (Crossref, then Semantic Scholar).

Usage:
  python3 scripts/fetch_literature_md.py --target 600 --out literature_corpus
"""
import argparse
import json
import os
import re
import time

import requests

API = "https://api.openalex.org"
FACULTY_WORK_ID = "W7171375780"  # Precision Investing (Padmakumari et al., 2026)

QUERIES = [
    "stock market prediction sentiment",
    "stock price prediction deep learning",
    "stock forecasting LSTM GRU",
    "financial sentiment analysis stock",
    "stock market news sentiment",
    "social media sentiment stock prediction",
    "stock return prediction machine learning",
    "financial time series forecasting neural network",
    "stock market movement prediction",
    "investor sentiment stock market",
    "textual sentiment financial markets",
    "stock price prediction attention mechanism",
    "volatility forecasting deep learning",
    "stock prediction transformer",
]

# Core finance tokens — non-exempt papers must match one in title+abstract+topics.
# NOTE: invest(or|ment)s?|investing only — bare "invest\w*" would match the verb
# "investigated" and admit off-domain papers (hate speech, drought, HCI studies).
CORE_RE = re.compile(
    r"\bstocks?\b|equit\w*|financ\w*|finans\w*|invest(or|ment)s?|investing|portfolio|crypt\w*|bitcoin|"
    r"forex|\bassets?\b|\btrading\b|\bbonds?\b|share price|closing price|index price|"
    r"market index|stock index|exchange rate|stock exchange|nasdaq|nepse|bist|masi|lq45|fintech",
    re.I,
)


def get(url, params=None, retries=4):
    headers = {"Accept-Encoding": "gzip, deflate"}  # avoid brotli decode bugs
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params = dict(params or {})
        params["mailto"] = mailto
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers=headers)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** i + 1)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
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


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]
    return s or "paper"


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def work_to_md(w, source):
    title = w.get("title") or "Untitled"
    authors = [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])]
    year = w.get("publication_year")
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    doi = w.get("doi") or ""
    oa = w.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    cited = w.get("cited_by_count", 0)
    topics = [t.get("display_name", "") for t in (w.get("topics") or [])[:6]]
    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
    short = w["id"].rsplit("/", 1)[-1]

    fm = {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "openalex_id": short,
        "cited_by": cited,
        "oa_url": oa_url,
        "is_oa": bool(oa.get("is_oa")),
        "topics": topics,
        "seed": bool(w.get("_seed")),
        "source": source,
    }

    lines = [
        "---",
        json.dumps(fm, indent=2, ensure_ascii=False),
        "---",
        "",
        f"# {title}",
        "",
        f"**Authors:** {', '.join(a for a in authors if a) or 'n/a'}  ",
        f"**Year:** {year} · **Venue:** {venue or 'n/a'} · **Cited by:** {cited}  ",
        f"**DOI:** {doi or 'n/a'} · **OpenAlex:** https://openalex.org/{short}  ",
        f"**Open access:** {oa_url or 'no'}  ",
        "",
    ]
    if topics:
        lines += ["**Topics:** " + "; ".join(topics), ""]
    lines += ["## Abstract", ""]
    lines += [abstract if abstract else "_No abstract available via OpenAlex._", ""]
    lines += ["## Links", "", f"- OpenAlex: https://openalex.org/{short}"]
    if doi:
        lines.append(f"- DOI: https://doi.org/{doi}")
    if oa_url:
        lines.append(f"- Open access: {oa_url}")
    return "\n".join(lines) + "\n", fm, abstract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--out", default="literature_corpus")
    ap.add_argument("--seed", default=FACULTY_WORK_ID)
    ap.add_argument("--from-year", type=int, default=2018)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, ".gitkeep"), "a").close()

    works = {}          # wid -> work record
    provenance = {}     # wid -> source string
    seen_titles = {}    # normalized title -> wid
    stats = {"dropped_dupe_id": 0, "dropped_dupe_title": 0, "dropped_offdomain": 0}

    def add_work(w, source, is_seed=False, exempt_guard=False):
        wid = w["id"].rsplit("/", 1)[-1]
        if wid in works:
            stats["dropped_dupe_id"] += 1
            return False
        nt = norm_title(w.get("title"))
        if nt and nt != "untitled" and nt in seen_titles:
            stats["dropped_dupe_title"] += 1
            return False
        if not exempt_guard:
            hay = " ".join([w.get("title") or "", reconstruct_abstract(w.get("abstract_inverted_index")),
                            " ".join(t.get("display_name", "") for t in (w.get("topics") or []))])
            if not CORE_RE.search(hay):
                stats["dropped_offdomain"] += 1
                return False
        w["_seed"] = is_seed or w.get("_seed", False)
        works[wid] = w
        provenance[wid] = source
        if nt:
            seen_titles[nt] = wid
        return True

    # 1) Faculty paper record -> seed references + related works
    faculty = get(f"{API}/works/{args.seed}")
    ref_ids = [u.rsplit("/", 1)[-1] for u in faculty.get("referenced_works", [])]
    rel_ids = [u.rsplit("/", 1)[-1] for u in faculty.get("related_works", [])]
    print(f"[faculty] refs={len(ref_ids)} related={len(rel_ids)}")

    def batch_add(ids, source, is_seed=False, exempt_guard=False):
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            data = get(f"{API}/works", {"filter": "openalex_id:" + "|".join(chunk), "per-page": 50})
            for w in data.get("results", []):
                add_work(w, source, is_seed=is_seed, exempt_guard=exempt_guard)
            time.sleep(0.15)

    batch_add(ref_ids, f"seed_ref:{args.seed}", is_seed=True, exempt_guard=True)
    batch_add(rel_ids, f"related:{args.seed}", exempt_guard=True)
    print(f"[after seeds+related] corpus={len(works)}")

    # 2) Lineage: works citing each seed
    base_filter = f"has_abstract:true,is_retracted:false,from_publication_date:{args.from_year}-01-01"
    for wid in ref_ids:
        data = get(f"{API}/works", {"filter": f"cites:{wid},{base_filter}", "per-page": 50})
        for w in data.get("results", []):
            if len(works) >= args.target:
                break
            add_work(w, f"cites:{wid}")
        print(f"[cites {wid}] corpus={len(works)}")
        time.sleep(0.15)
        if len(works) >= args.target:
            break

    # 3) Topic backfill via title_and_abstract.search (tight relevance)
    for q in QUERIES:
        if len(works) >= args.target:
            break
        for sort in (None, "cited_by_count:desc"):
            if len(works) >= args.target:
                break
            params = {
                "filter": f"title_and_abstract.search:{q},{base_filter}",
                "per-page": 200,
            }
            if sort:
                params["sort"] = sort
            data = get(f"{API}/works", params)
            for w in data.get("results", []):
                if len(works) >= args.target:
                    break
                add_work(w, f"search:{q}")
        print(f"[search '{q}'] corpus={len(works)}")

    # 4) Write markdown + manifest
    entries = []
    written = 0
    for wid, w in works.items():
        md, fm, abstract = work_to_md(w, provenance.get(wid, "unknown"))
        first_author = (w.get("authorships") or [{}])[0].get("author", {}).get("display_name", "unknown")
        last_name = first_author.split()[-1] if first_author else "unknown"
        fname = f"{fm['year'] or 'nd'}_{slug(last_name)}_{wid}.md"
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(md)
        fm["file"] = fname
        fm["abstract_words"] = len(abstract.split())
        entries.append(fm)
        written += 1
        if written % 50 == 0:
            print(f"[write] {written} files")

    years = sorted({e["year"] for e in entries if e["year"]})
    src_counts = {}
    for p in provenance.values():
        key = p.split(":")[0]
        src_counts[key] = src_counts.get(key, 0) + 1
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "OpenAlex (https://api.openalex.org)",
        "seed_work": f"{args.seed} — Precision Investing (Padmakumari et al., 2026)",
        "seed_references": len(ref_ids),
        "faculty_related": len(rel_ids),
        "queries": QUERIES,
        "filters": base_filter,
        "target": args.target,
        "total_papers": written,
        "by_source": src_counts,
        "dropped": stats,
        "year_range": [years[0], years[-1]] if years else None,
        "with_abstract": sum(1 for e in entries if e["abstract_words"] > 0),
        "open_access": sum(1 for e in entries if e["is_oa"]),
        "entries": entries,
    }
    with open(os.path.join(args.out, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "_manifest.txt"), "w", encoding="utf-8") as f:
        f.write(f"Literature corpus manifest — {written} papers\n")
        f.write(f"Source: OpenAlex · generated {manifest['generated_at']}\n")
        f.write(f"Year range: {manifest['year_range']} · with abstract: {manifest['with_abstract']} · OA: {manifest['open_access']}\n")
        f.write(f"By source: {src_counts}\n\n")
        f.write(f"{'file':<70} {'year':<6} {'venue':<40} {'cited':>6}\n")
        f.write("-" * 130 + "\n")
        for e in sorted(entries, key=lambda x: (x["year"] or 0, x["title"] or "")):
            f.write(f"{e['file']:<70} {str(e['year'] or ''):<6} {(e['venue'] or '')[:40]:<40} {e['cited_by']:>6}\n")

    print(json.dumps({k: manifest[k] for k in ("total_papers", "by_source", "dropped", "year_range", "with_abstract", "open_access")}, indent=2))


if __name__ == "__main__":
    main()

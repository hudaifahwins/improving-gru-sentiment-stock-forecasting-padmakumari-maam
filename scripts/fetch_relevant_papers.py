#!/usr/bin/env python3
"""Targeted literature expansion: fetch accessible paper records for this project.

Why this exists
---------------
`literature_corpus/` (575 papers) is anchored to the faculty paper's citation
lineage. This script adds a *method-driven* layer: papers covering the specific
gaps our audit identified — cross-sectional/panel return forecasting,
evaluation rigor (multiple testing, backtests, transaction costs),
firm-level and FinBERT sentiment, volatility/regime conditioning, and the
GRU/LSTM baseline literature.

Accessibility rules (enforced, not assumed)
-------------------------------------------
- Only records with an OpenAlex abstract are kept (`has_abstract:true`) and
  the reconstructed abstract is stored verbatim; no text is generated.
- Retracted works are excluded (`is_retracted:false`).
- Open-access status, OA landing URL and OA PDF URL are recorded per paper so
  the accessible subset is explicit. Full-text PDFs are NOT bulk-downloaded
  (most `oa_url` values are publisher landing pages); `--download-oa-pdfs`
  fetches only direct `.pdf` links, for a capped number of papers.

Outputs (new folder; the audited 575-paper corpus is never modified)
--------------------------------------------------------------------
  <out>/paper_<year>_<slug>_<openalex_id>.md   one file per paper
  <out>/_manifest.json                         full machine-readable records
  <out>/SUMMARY.md                             counts, themes, top papers
  <out>/_dropped.json                          dedupe / guard / error log

Usage
-----
    python3 scripts/fetch_relevant_papers.py --target 1500
    python3 scripts/fetch_relevant_papers.py --target 300 --themes sentiment
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

API = "https://api.openalex.org"
SELECT = (
    "id,doi,title,publication_year,publication_date,type,primary_location,"
    "authorships,cited_by_count,open_access,abstract_inverted_index,topics,"
    "primary_topic"
)

# ---------------------------------------------------------------------------
# Query plan: theme -> queries. Each query is run under two sort orders
# (relevance default, then most-cited) since OpenAlex pages only 200/query.
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "forecasting_models": [
        "gated recurrent unit stock price prediction",
        "GRU LSTM stock market forecasting",
        "recurrent neural network stock return forecasting",
        "deep learning stock price prediction",
        "transformer stock price forecasting",
        "attention mechanism stock prediction",
        "graph neural network stock prediction",
        "temporal convolutional network financial forecasting",
        "hybrid LSTM GRU stock prediction",
        "machine learning stock market prediction comparison",
    ],
    "targets_and_evaluation": [
        "stock return prediction machine learning",
        "cross-sectional stock return prediction",
        "stock ranking learning to rank returns",
        "directional accuracy stock forecasting evaluation",
        "forecast evaluation diebold mariano financial",
        "backtesting overfitting trading strategy",
        "multiple testing financial machine learning",
        "walk-forward validation stock forecasting",
        "look-ahead bias financial machine learning",
        "data snooping trading strategy",
        "transaction costs machine learning portfolio",
        "sharpe ratio machine learning trading strategy",
        "out-of-sample return predictability",
    ],
    "sentiment": [
        "financial sentiment analysis stock returns",
        "news sentiment stock market prediction",
        "FinBERT financial sentiment",
        "earnings call sentiment stock returns",
        "SEC filing sentiment return prediction",
        "social media sentiment stock returns",
        "GDELT news sentiment financial markets",
        "investor sentiment return predictability",
        "textual analysis stock returns",
        "sentiment volatility interaction stock",
        "large language model financial forecasting",
        "annual report tone stock returns",
        "analyst forecast sentiment stock",
        "earnings announcement abnormal returns prediction",
        "news sentiment intraday stock returns",
        "financial news event study machine learning",
    ],
    "volatility_and_regime": [
        "realized volatility forecasting machine learning",
        "GARCH neural network hybrid volatility forecasting",
        "volatility regime stock returns machine learning",
        "uncertainty stock return prediction",
        "volatility spillover machine learning",
        "regime switching neural network finance",
    ],
    "portfolio_and_microstructure": [
        "portfolio optimization machine learning returns",
        "transaction cost portfolio optimization deep learning",
        "limit order book deep learning",
        "market microstructure machine learning",
        "asset pricing neural network",
        "conformal prediction financial time series",
        "quantile regression stock returns",
    ],
    "markets": [
        "emerging market stock prediction deep learning",
        "Korean stock market forecasting",
        "cross-market stock prediction transfer learning",
        "index stock forecasting comparison markets",
        "panel data stock return prediction",
        "cross-sectional momentum machine learning",
    ],
}

# Finance-domain guard (same core as fetch_literature_md.py, plus evaluation
# vocabulary so rigorous backtest/econometrics papers are not dropped).
CORE_RE = re.compile(
    r"\bstocks?\b|equit\w*|financ\w*|finans\w*|invest(or|ment)s?|investing|portfolio|"
    r"crypt\w*|bitcoin|forex|\bassets?\b|\btrading\b|\bbonds?\b|share price|"
    r"closing price|index price|market index|stock index|exchange rate|stock exchange|"
    r"nasdaq|nepse|bist|masi|lq45|fintech|asset pricing|backtest\w*|"
    r"return predictability|factor model|sharpe|volatil\w*|derivative\w*|"
    r"\boptions?\b|futures|earnings|dividend\w*|valuation",
    re.I,
)

# Title-level finance vocabulary (strong evidence of a finance subject).
TITLE_FIN_RE = re.compile(
    r"\bstocks?\b|equit(y|ies)|financ\w*|invest\w*|portfolio|trading|trader|"
    r"crypt\w*|bitcoin|forex|\bbonds?\b|\bassets?\b|volatil\w*|\breturns?\b|"
    r"\bmarkets?\b|sharpe|hedg\w*|\boptions?\b|futures|earnings|dividend\w*|"
    r"valuation|nasdaq|nepse|bist|masi|lq45",
    re.I,
)

# Word-sense traps: "carbon stock", "livestock", soil/biomass papers, etc.
OFF_DOMAIN_RE = re.compile(
    r"carbon stock|soil|biomass|mangrove|livestock|drought|crop\b|forest\w*|"
    r"clinical|patient\w*|hate speech|protein|gene\b|water quality|air quality|"
    r"\bmarketing\b|touris\w*|supply chain|agricultur\w*|ecosystem services",
    re.I,
)

FIN_FIELDS = {
    "economics, econometrics and finance",
    "business, management and accounting",
    "decision sciences",
    "finance",
}


def looks_finance(title: str, abstract: str, field: str) -> bool:
    """Two-tier relevance test: title evidence, or finance field + domain text."""
    if OFF_DOMAIN_RE.search(title or ""):
        return False
    if TITLE_FIN_RE.search(title or ""):
        return True
    field_l = (field or "").strip().lower()
    if field_l in FIN_FIELDS and CORE_RE.search(f"{title} {abstract}"):
        return True
    return False


def get(url: str, params: dict, retries: int = 10) -> dict:
    """GET with patient backoff — OpenAlex throttles aggressively without a key."""
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "research-corpus-builder/1.0 (academic use)",
    }
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        params = dict(params, mailto=mailto)
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60, headers=headers)
            if r.status_code == 429:
                wait = min(90.0, 5.0 * (2 ** i)) + random.uniform(0, 2.0)
                print(f"[429] backing off {wait:.0f}s (attempt {i + 1}/{retries})",
                      flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(60.0, 2 ** i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(min(60.0, 2 ** i))
    raise RuntimeError("OpenAlex unreachable after retries (rate limit)")


def reconstruct_abstract(inv: dict | None) -> str:
    """Rebuild verbatim abstract text from OpenAlex's inverted index."""
    if not inv:
        return ""
    words: list[str] = []
    for word, positions in inv.items():
        for p in positions:
            if p >= len(words):
                words.extend([""] * (p - len(words) + 1))
            words[p] = word
    return " ".join(w for w in words if w)


def norm_title(t: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return s[:n] or "paper"


# When a paper matches queries from several themes, label it with the most
# specific one (left wins).
THEME_PRIORITY = [
    "sentiment",
    "targets_and_evaluation",
    "portfolio_and_microstructure",
    "volatility_and_regime",
    "markets",
    "forecasting_models",
]


def load_existing(out: Path) -> tuple[set[str], set[str]]:
    """OpenAlex IDs and normalized titles already in any local corpus folder."""
    ids: set[str] = set()
    titles: set[str] = set()
    roots = [Path("literature_corpus"), Path("data")]
    for root in roots:
        if not root.exists():
            continue
        for path in list(root.glob("*.md")) + list(root.glob("*/*.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.startswith("---"):
                continue
            block = text.split("---", 2)[1]
            try:
                fm = json.loads(block)
            except json.JSONDecodeError:
                continue
            if fm.get("openalex_id"):
                ids.add(str(fm["openalex_id"]))
            if fm.get("title"):
                titles.add(norm_title(fm["title"]))
    for path in list(Path("data").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("openalex_id"):
                ids.add(str(rec["openalex_id"]))
            if rec.get("title"):
                titles.add(norm_title(rec["title"]))
    return ids, titles


def record_from_work(w: dict, theme: str, query: str) -> dict:
    wid = w["id"].rsplit("/", 1)[-1]
    oa = w.get("open_access") or {}
    best = oa.get("best_oa_location") or {}
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    ptopic = w.get("primary_topic") or {}
    field = ((ptopic.get("field") or {}).get("display_name") or "")
    return {
        "openalex_id": wid,
        "doi": w.get("doi") or "",
        "title": w.get("title") or "Untitled",
        "authors": [
            a.get("author", {}).get("display_name", "")
            for a in (w.get("authorships") or [])
        ],
        "year": w.get("publication_year"),
        "publication_date": w.get("publication_date") or "",
        "venue": src.get("display_name") or "",
        "type": w.get("type") or "",
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "is_oa": bool(oa.get("is_oa")),
        "oa_status": oa.get("oa_status") or "",
        "oa_url": oa.get("oa_url") or "",
        "oa_pdf_url": (best.get("pdf_url") or loc.get("pdf_url") or ""),
        "cited_by_count": w.get("cited_by_count") or 0,
        "topics": [t.get("display_name", "") for t in (w.get("topics") or [])[:6]],
        "primary_topic": ptopic.get("display_name") or "",
        "primary_field": field,
        "primary_domain": ((ptopic.get("domain") or {}).get("display_name") or ""),
        "theme": theme,
        "source_query": query,
    }


def paper_md(rec: dict) -> str:
    fm = {
        k: rec[k]
        for k in (
            "title", "authors", "year", "venue", "doi", "openalex_id",
            "cited_by_count", "oa_url", "is_oa", "topics", "theme",
            "source_query",
        )
    }
    fm["access"] = "abstract + open-access link" if rec["is_oa"] else "abstract only"
    lines = [
        "---",
        json.dumps(fm, indent=2, ensure_ascii=False),
        "---",
        "",
        f"# {rec['title']}",
        "",
        f"**Authors:** {', '.join(a for a in rec['authors'] if a) or 'n/a'}  ",
        f"**Year:** {rec['year']} · **Venue:** {rec['venue'] or 'n/a'} · "
        f"**Cited by:** {rec['cited_by_count']}  ",
        f"**DOI:** {rec['doi'] or 'n/a'} · "
        f"**OpenAlex:** https://openalex.org/{rec['openalex_id']}  ",
        f"**Open access:** {rec['oa_url'] or 'no'}  ",
        f"**Theme:** {rec['theme']}  ",
        "",
    ]
    if rec["topics"]:
        lines += ["**Topics:** " + "; ".join(rec["topics"]), ""]
    lines += ["## Abstract", "", rec["abstract"] or "_No abstract available._", ""]
    lines += ["## Links", "", f"- OpenAlex: https://openalex.org/{rec['openalex_id']}"]
    if rec["doi"]:
        lines.append(f"- DOI: https://doi.org/{rec['doi']}")
    if rec["oa_url"]:
        lines.append(f"- Open access: {rec['oa_url']}")
    if rec["oa_pdf_url"]:
        lines.append(f"- OA PDF: {rec['oa_pdf_url']}")
    return "\n".join(lines) + "\n"


def write_summary(out: Path, records: list[dict], stats: dict) -> None:
    by_theme = defaultdict(list)
    for r in records:
        by_theme[r["theme"]].append(r)
    years = Counter(r["year"] for r in records if r["year"])
    oa = sum(1 for r in records if r["is_oa"])
    with_abs = sum(1 for r in records if r["abstract"].strip())

    lines = [
        "# Relevant-paper expansion — method-driven corpus",
        "",
        f"Fetched from OpenAlex on {time.strftime('%Y-%m-%d')}. "
        "Abstracts are verbatim reconstructions of the OpenAlex inverted index; "
        "no bibliographic field is generated.",
        "",
        "## Accounting",
        "",
        f"- Papers retained: **{len(records)}**",
        f"- With abstract: **{with_abs}/{len(records)}**",
        f"- Open access: **{oa}/{len(records)}** ({oa / max(len(records), 1):.1%})",
        f"- Candidate records seen: {stats['candidates']} · "
        f"duplicate of existing corpus: {stats['dropped_existing']} · "
        f"off-domain guard: {stats['dropped_guard']} · "
        f"duplicate within run: {stats['dropped_dupe']} · "
        f"errors: {stats['errors']}",
        "",
        "## By theme",
        "",
        "| Theme | Papers | Open access | With abstract |",
        "|---|---|---|---|",
    ]
    for theme, items in sorted(by_theme.items(), key=lambda kv: -len(kv[1])):
        lines.append(
            f"| {theme} | {len(items)} | {sum(1 for r in items if r['is_oa'])} | "
            f"{sum(1 for r in items if r['abstract'].strip())} |"
        )
    lines += ["", "## By year", ""]
    lines.append(" · ".join(f"{y}: {n}" for y, n in sorted(years.items())))
    lines += ["", "## Most-cited retained papers", ""]
    for r in sorted(records, key=lambda r: -r["cited_by_count"])[:20]:
        lines.append(
            f"- {r['year']} — {r['title']} ({r['venue'] or 'n/a'}, "
            f"cited {r['cited_by_count']}) — {r['theme']}"
            + (f" — {r['oa_url']}" if r["oa_url"] else "")
        )
    lines += [
        "",
        "## Limits",
        "",
        "- Abstract-level records only. Full texts are frequently paywalled; "
        "`oa_url`/`oa_pdf_url` mark the accessible subset.",
        "- Retrieval is query-based, so recall is bounded by the query plan; "
        "this folder complements rather than replaces the citation-lineage corpus.",
        "- Citation counts are OpenAlex values at fetch time.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500,
                    help="max new papers to retain after ranking")
    ap.add_argument("--out", default="literature_corpus/relevant_2026-11-26")
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--per-query", type=int, default=200,
                    help="OpenAlex page size per query (max 200)")
    ap.add_argument("--pages", type=int, default=2,
                    help="OpenAlex result pages per query (200 records each)")
    ap.add_argument("--themes", default="", help="comma-separated theme filter")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--download-oa-pdfs", action="store_true",
                    help="also download direct .pdf OA links (capped by --max-pdfs)")
    ap.add_argument("--max-pdfs", type=int, default=100)
    args = ap.parse_args()

    selected = [t.strip() for t in args.themes.split(",") if t.strip()]
    plan = {t: qs for t, qs in THEMES.items() if not selected or t in selected}
    if not plan:
        raise SystemExit(f"no matching themes; available: {', '.join(THEMES)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitkeep").touch()
    for stale in out.glob("paper_*.md"):
        stale.unlink()

    existing_ids, existing_titles = load_existing(out)
    print(f"[dedupe] {len(existing_ids)} existing OpenAlex IDs, "
          f"{len(existing_titles)} existing titles")

    stats = {
        "candidates": 0, "dropped_existing": 0, "dropped_guard": 0,
        "dropped_dupe": 0, "errors": 0,
    }
    dropped: list[dict] = []
    works: dict[str, dict] = {}
    hits: Counter = Counter()
    theme_hits: dict[str, set[str]] = defaultdict(set)

    base = (
        f"has_abstract:true,is_retracted:false,"
        f"from_publication_date:{args.from_year}-01-01"
    )

    for theme, queries in plan.items():
        for q in queries:
            for page in range(1, args.pages + 1):
                params = {
                    "filter": f"title_and_abstract.search:{q},{base}",
                    "per-page": args.per_query,
                    "page": page,
                    "select": SELECT,
                }
                try:
                    data = get(f"{API}/works", params)
                except Exception as exc:  # keep going; report at the end
                    stats["errors"] += 1
                    dropped.append({"query": q, "page": page, "reason": f"error: {exc}"})
                    print(f"[error] {q!r} (page {page}): {exc}")
                    continue

                added = 0
                for w in data.get("results", []):
                    stats["candidates"] += 1
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    if not wid:
                        continue
                    nt = norm_title(w.get("title"))
                    if wid in existing_ids or (nt and nt in existing_titles):
                        stats["dropped_existing"] += 1
                        continue
                    abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
                    ptopic = w.get("primary_topic") or {}
                    field = ((ptopic.get("field") or {}).get("display_name") or "")
                    if not looks_finance(w.get("title") or "", abstract, field):
                        stats["dropped_guard"] += 1
                        dropped.append({"id": wid, "title": w.get("title"),
                                        "reason": "offdomain", "query": q})
                        continue
                    if wid in works:
                        stats["dropped_dupe"] += 1
                        hits[wid] += 1
                        theme_hits[wid].add(theme)
                        continue
                    rec = record_from_work(w, theme, q)
                    if not rec["abstract"].strip():
                        stats["dropped_guard"] += 1
                        continue
                    works[wid] = rec
                    hits[wid] = 1
                    theme_hits[wid].add(theme)
                    added += 1
                print(f"[{theme}] {q!r} (page {page}) -> "
                      f"+{added} (total {len(works)})", flush=True)
                time.sleep(args.sleep)
                if len(data.get("results", [])) < args.per_query:
                    break  # query exhausted; no deeper pages exist

    # Rank: multi-query corroboration first, then citations, then recency.
    ranked = sorted(
        works.values(),
        key=lambda r: (
            hits[r["openalex_id"]],
            math.log1p(r["cited_by_count"]),
            r["year"] or 0,
        ),
        reverse=True,
    )[: args.target]

    order = {t: i for i, t in enumerate(THEME_PRIORITY)}
    for rec in ranked:
        themes = theme_hits.get(rec["openalex_id"]) or {rec["theme"]}
        rec["theme"] = sorted(themes, key=lambda t: order.get(t, 99))[0]

    ids_kept, titles_kept = set(), set()
    final: list[dict] = []
    for rec in ranked:
        nt = norm_title(rec["title"])
        if rec["openalex_id"] in ids_kept or (nt and nt in titles_kept):
            continue
        ids_kept.add(rec["openalex_id"])
        titles_kept.add(nt)
        final.append(rec)

    for rec in final:
        name = (f"paper_{rec['year'] or 'nd'}_{slug(rec['title'])}"
                f"_{rec['openalex_id']}.md")
        (out / name).write_text(paper_md(rec), encoding="utf-8")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": API,
        "themes": {t: qs for t, qs in plan.items()},
        "filters": {
            "pages_per_query": args.pages,
            "has_abstract": True,
            "is_retracted": False,
            "from_publication_date": f"{args.from_year}-01-01",
            "domain_guard": CORE_RE.pattern,
            "title_guard": TITLE_FIN_RE.pattern,
            "off_domain_guard": OFF_DOMAIN_RE.pattern,
            "finance_fields": sorted(FIN_FIELDS),
        },
        "stats": stats,
        "target": args.target,
        "retained": len(final),
        "papers": final,
    }
    (out / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "_dropped.json").write_text(
        json.dumps(dropped, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary(out, final, stats)

    if args.download_oa_pdfs:
        pdf_dir = out / "pdfs"
        pdf_dir.mkdir(exist_ok=True)
        got = 0
        for rec in final:
            if got >= args.max_pdfs or not rec["oa_pdf_url"].lower().endswith(".pdf"):
                continue
            path = pdf_dir / f"{rec['openalex_id']}.pdf"
            try:
                r = requests.get(rec["oa_pdf_url"], timeout=30, stream=True,
                                 headers={"User-Agent": "research-corpus-builder/1.0"})
                if r.status_code == 200 and r.headers.get(
                        "Content-Type", "").lower().startswith("application/pdf"):
                    path.write_bytes(r.content)
                    got += 1
                    print(f"[pdf] {path.name} ({path.stat().st_size // 1024} KB)")
            except requests.RequestException:
                pass
            time.sleep(0.2)
        print(f"[pdf] downloaded {got} PDFs")

    print(f"\n[done] retained {len(final)} new papers in {out}")
    print(f"       abstracts {sum(1 for r in final if r['abstract'].strip())}/{len(final)}"
          f" · open access {sum(1 for r in final if r['is_oa'])}/{len(final)}")
    print(f"       dropped: existing {stats['dropped_existing']}, "
          f"guard {stats['dropped_guard']}, dupe {stats['dropped_dupe']}, "
          f"errors {stats['errors']}")


if __name__ == "__main__":
    main()

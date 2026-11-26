#!/usr/bin/env python3
"""Fetch accessible arXiv preprints relevant to this project.

OpenAlex/Semantic Scholar were rate-limiting (HTTP 429) during the corpus
build, so this script uses the arXiv API — a genuinely accessible source:
every record gets a public abstract page AND a direct PDF link.

Relevance is restricted to the project's method gaps: stock/return
forecasting, cross-sectional prediction, financial sentiment (FinBERT, news,
filings), volatility, backtesting/overfitting, portfolio optimisation, limit
order books, and trading strategies.

Outputs (separate folder; other corpora are untouched):
  <out>/paper_<year>_<slug>_<arxiv_id>.md
  <out>/_manifest.json
  <out>/SUMMARY.md

Usage:
  python3 scripts/fetch_arxiv_papers.py --target 400
  python3 scripts/fetch_arxiv_papers.py --download-pdfs --max-pdfs 50
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API = "https://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

QUERIES = [
    'all:"stock prediction" AND all:"deep learning"',
    'all:"stock return prediction"',
    'all:"cross-sectional" AND all:"return prediction"',
    'all:"financial sentiment" AND all:"stock"',
    'all:"news sentiment" AND all:"stock"',
    'all:"FinBERT"',
    'all:"earnings call" AND all:sentiment',
    'all:"10-K" AND all:sentiment',
    'all:"volatility forecasting" AND all:"machine learning"',
    'all:"backtesting" AND all:"overfitting"',
    'all:"multiple testing" AND all:"finance"',
    'all:"portfolio optimization" AND all:"deep learning"',
    'all:"limit order book" AND all:"deep learning"',
    'all:"trading strategy" AND all:"reinforcement learning"',
    'all:"financial time series" AND all:"transformer"',
    'all:"large language model" AND all:"financial forecasting"',
    'all:"regime switching" AND all:"stock"',
    'all:"anomalies" AND all:"machine learning" AND all:"returns"',
]

CORE_RE = re.compile(
    r"\bstocks?\b|equit\w*|financ\w*|invest\w*|portfolio|crypt\w*|bitcoin|forex|"
    r"\bassets?\b|trading|trader|\bbonds?\b|share price|market index|stock index|"
    r"nasdaq|nepse|bist|masi|asset pricing|backtest\w*|sharpe|volatil\w*|"
    r"return predictability|factor model|earnings|dividend\w*|valuation|"
    r"order book|market microstructure|hedg\w*",
    re.I,
)
TITLE_FIN_RE = re.compile(
    r"\bstocks?\b|equit(y|ies)|financ\w*|invest\w*|portfolio|trading|trader|"
    r"crypt\w*|bitcoin|forex|\bbonds?\b|\bassets?\b|volatil\w*|\breturns?\b|"
    r"\bmarkets?\b|sharpe|hedg\w*|earnings|dividend\w*|valuation|"
    r"order book|price prediction",
    re.I,
)
OFF_DOMAIN_RE = re.compile(
    r"carbon stock|soil|biomass|mangrove|livestock|drought|\bcrop\b|forest\w*|"
    r"clinical|patient\w*|hate speech|protein|\bgene\b|water quality|air quality|"
    r"\bmarketing\b|touris\w*|supply chain|agricultur\w*|ecosystem services",
    re.I,
)

THEME_RULES = [
    ("sentiment", r"sentiment|\btone\b|news|finbert|textual|10-k|earnings call|llm|language model"),
    ("targets_and_evaluation", r"cross-section|panel|backtest|multiple testing|overfit|transaction cost|out-of-sample|return predictability|anomal"),
    ("portfolio_and_microstructure", r"portfolio|order book|microstructure|asset pricing|execution"),
    ("volatility_and_regime", r"volatil|garch|regime"),
    ("forecasting_models", r"."),
]


def get(params: dict, retries: int = 4) -> str:
    for i in range(retries):
        try:
            r = requests.get(API, params=params, timeout=60,
                             headers={"User-Agent": "research-corpus-builder/1.0"})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(5 * (i + 1))
    raise RuntimeError("arXiv unreachable")


def norm_title(t: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def slug(s: str, n: int = 40) -> str:
    return (re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:n] or "paper")


def theme_for(text: str) -> str:
    for name, pat in THEME_RULES:
        if re.search(pat, text, re.I):
            return name
    return "forecasting_models"


def existing_keys() -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    titles: set[str] = set()
    for root in (Path("literature_corpus"), Path("data")):
        if not root.exists():
            continue
        for path in list(root.glob("*.md")) + list(root.glob("*/*.md")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.startswith("---"):
                continue
            try:
                fm = json.loads(text.split("---", 2)[1])
            except (json.JSONDecodeError, IndexError):
                continue
            for key in ("openalex_id", "arxiv_id"):
                if fm.get(key):
                    ids.add(str(fm[key]).split("v")[0])
            if fm.get("title"):
                titles.add(norm_title(fm["title"]))
    for path in Path("data").glob("*.jsonl"):
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


def parse(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", default="", namespaces=NS) or "").strip()
        aid = raw_id.rsplit("/", 1)[-1]
        base = re.sub(r"v\d+$", "", aid)
        pdf = ""
        alt = ""
        for link in e.findall("a:link", NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf = link.get("href") or ""
            elif link.get("rel") == "alternate":
                alt = link.get("href") or ""
        out.append({
            "arxiv_id": base,
            "title": " ".join((e.findtext("a:title", default="", namespaces=NS) or "").split()),
            "authors": [
                (a.findtext("a:name", default="", namespaces=NS) or "").strip()
                for a in e.findall("a:author", NS)
            ],
            "abstract": " ".join((e.findtext("a:summary", default="", namespaces=NS) or "").split()),
            "published": (e.findtext("a:published", default="", namespaces=NS) or "")[:10],
            "year": int((e.findtext("a:published", default="2020", namespaces=NS) or "2020")[:4]),
            "primary_category": (e.findtext("arxiv:primary_category", default="", namespaces=NS) or ""),
            "journal_ref": (e.findtext("arxiv:journal_ref", default="", namespaces=NS) or ""),
            "doi": (e.findtext("arxiv:doi", default="", namespaces=NS) or ""),
            "abs_url": alt or f"https://arxiv.org/abs/{base}",
            "pdf_url": pdf or f"https://arxiv.org/pdf/{base}.pdf",
        })
    return out


def paper_md(rec: dict) -> str:
    fm = {k: rec[k] for k in (
        "title", "authors", "year", "venue", "doi", "arxiv_id", "primary_category",
        "journal_ref", "abs_url", "pdf_url", "theme", "source_query",
    )}
    fm["access"] = "open access (arXiv abstract + PDF)"
    lines = [
        "---", json.dumps(fm, indent=2, ensure_ascii=False), "---", "",
        f"# {rec['title']}", "",
        f"**Authors:** {', '.join(a for a in rec['authors'] if a) or 'n/a'}  ",
        f"**Year:** {rec['year']} · **arXiv:** {rec['primary_category'] or 'n/a'}"
        f" · **Journal ref:** {rec['journal_ref'] or 'n/a'}  ",
        f"**DOI:** {rec['doi'] or 'n/a'} · **Theme:** {rec['theme']}  ",
        f"**Open access:** {rec['abs_url']}  ",
        "",
        "## Abstract", "", rec["abstract"] or "_No abstract available._", "",
        "## Links", "",
        f"- Abstract: {rec['abs_url']}",
        f"- PDF: {rec['pdf_url']}",
    ]
    if rec["doi"]:
        lines.append(f"- DOI: https://doi.org/{rec['doi']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--per-query", type=int, default=100)
    ap.add_argument("--out", default="literature_corpus/arxiv_2026-11-26")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="arXiv asks for ~3s between API calls")
    ap.add_argument("--download-pdfs", action="store_true")
    ap.add_argument("--max-pdfs", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitkeep").touch()
    for stale in out.glob("paper_*.md"):
        stale.unlink()

    ids, titles = existing_keys()
    print(f"[dedupe] {len(ids)} ids / {len(titles)} titles already present")

    works: dict[str, dict] = {}
    stats = {"candidates": 0, "dropped_existing": 0, "dropped_guard": 0,
             "dropped_dupe": 0, "errors": 0}

    for q in QUERIES:
        if len(works) >= args.target:
            break
        try:
            xml_text = get({"search_query": q, "start": 0,
                            "max_results": args.per_query, "sortBy": "relevance"})
        except Exception as exc:
            stats["errors"] += 1
            print(f"[error] {q}: {exc}")
            continue
        added = 0
        for rec in parse(xml_text):
            stats["candidates"] += 1
            if rec["arxiv_id"] in ids or norm_title(rec["title"]) in titles:
                stats["dropped_existing"] += 1
                continue
            if rec["arxiv_id"] in works:
                stats["dropped_dupe"] += 1
                continue
            if OFF_DOMAIN_RE.search(rec["title"]):
                stats["dropped_guard"] += 1
                continue
            if not (TITLE_FIN_RE.search(rec["title"])
                    or CORE_RE.search(rec["title"] + " " + rec["abstract"])):
                stats["dropped_guard"] += 1
                continue
            rec["theme"] = theme_for(rec["title"] + " " + rec["abstract"])
            rec["source_query"] = q
            rec["venue"] = rec["journal_ref"] or "arXiv"
            works[rec["arxiv_id"]] = rec
            added += 1
        print(f"[arxiv] {q[:60]!r} -> +{added} (total {len(works)})", flush=True)
        time.sleep(args.sleep)

    ranked = sorted(works.values(), key=lambda r: r["year"], reverse=True)[: args.target]
    for rec in ranked:
        name = f"paper_{rec['year']}_{slug(rec['title'])}_{rec['arxiv_id']}.md"
        (out / name).write_text(paper_md(rec), encoding="utf-8")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": API,
        "queries": QUERIES,
        "stats": stats,
        "retained": len(ranked),
        "access": "arXiv abstract page + direct PDF for every record",
        "papers": ranked,
    }
    (out / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    years = {}
    themes = {}
    for r in ranked:
        years[r["year"]] = years.get(r["year"], 0) + 1
        themes[r["theme"]] = themes.get(r["theme"], 0) + 1
    lines = [
        "# arXiv supplement — accessible preprints",
        "",
        f"Fetched {time.strftime('%Y-%m-%d')} from the arXiv API. Every paper has an "
        "abstract page and a direct PDF link (fully accessible).",
        "",
        f"- Retained: **{len(ranked)}**",
        f"- Candidates seen: {stats['candidates']} · already present locally: "
        f"{stats['dropped_existing']} · relevance guard: {stats['dropped_guard']} · "
        f"intra-run duplicates: {stats['dropped_dupe']} · errors: {stats['errors']}",
        "",
        "| Theme | Papers |",
        "|---|---|",
    ]
    for t, n in sorted(themes.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {t} | {n} |")
    lines += ["", "Years: " + " · ".join(f"{y}: {n}" for y, n in sorted(years.items())), ""]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    if args.download_pdfs:
        pdf_dir = out / "pdfs"
        pdf_dir.mkdir(exist_ok=True)
        got = 0
        for rec in ranked:
            if got >= args.max_pdfs:
                break
            path = pdf_dir / f"{rec['arxiv_id']}.pdf"
            try:
                r = requests.get(rec["pdf_url"], timeout=60,
                                 headers={"User-Agent": "research-corpus-builder/1.0"})
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    path.write_bytes(r.content)
                    got += 1
                    print(f"[pdf] {path.name} ({len(r.content) // 1024} KB)")
            except requests.RequestException:
                pass
            time.sleep(1.0)
        print(f"[pdf] downloaded {got}")

    print(f"\n[done] {len(ranked)} arXiv papers in {out}")


if __name__ == "__main__":
    main()

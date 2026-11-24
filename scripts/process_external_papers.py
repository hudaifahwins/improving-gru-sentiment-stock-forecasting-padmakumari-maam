#!/usr/bin/env python3
"""Process data/external_200plus_papers.jsonl into a structured literature report.

Reads the 200 OpenAlex records fetched by scripts/fetch_external_200plus.py and
computes exact abstract-level keyword/regex counts.

Integrity rules:
  - No fabrication; only retrieved abstracts are scanned.
  - All findings are labeled abstract-only (or oa-fulltext where OA full text was
    retrieved); unmentioned dimensions are recorded as not-stated-in-abstract.

Outputs (local, gitignored data/ folder):
  - data/external_200plus_processed.json
  - data/external_200plus_analysis.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = {
    "gru": re.compile(r"\bgru\b", re.I),
    "lstm": re.compile(r"\blstm\b", re.I),
    "return_target": re.compile(r"\breturns?\b", re.I),
    "price_level": re.compile(r"\bprices?\b", re.I),
    "multi_day": re.compile(r"\bmulti[- ]day\b|\bhorizon", re.I),
    "leakage": re.compile(
        r"\bleakag\w*\b|\bwalk[- ]forward\w*|\bcross[- ]validation|\bholdout|\boos\b", re.I),
    "significance": re.compile(
        r"\bsignificanc\w*\b|\bstatist\w*\b|\bdiebold|\bwilcoxon|\bmcnemar|\bt[- ]test|\bperron|\bbootstrap", re.I),
    "news_sentiment": re.compile(r"\bnews\b|\bgdelt\b|\bsentiment\w*\b|\btwitter\b", re.I),
}

NOTABLE_TOPICS = ["multi_day", "leakage", "news_sentiment", "return_target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/external_200plus_papers.jsonl")
    ap.add_argument("--out-json", default="data/external_200plus_processed.json")
    ap.add_argument("--out-md", default="data/external_200plus_analysis.md")
    args = ap.parse_args()

    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n = len(records)
    years = [r.get("year") for r in records if r.get("year")]
    y_min, y_max = (min(years), max(years)) if years else (None, None)
    oa_n = sum(1 for r in records if r.get("is_oa"))
    abs_n = sum(1 for r in records if (r.get("abstract") or "").strip())

    counts = {}
    occ = {}
    notable = []
    for topic, rx in PATTERNS.items():
        recs_hit = 0
        occ_n = 0
        hit_records = []
        for r in records:
            text = r.get("abstract") or ""
            o = len(rx.findall(text))
            occ_n += o
            if o > 0:
                recs_hit += 1
                hit_records.append(r)
        counts[topic] = {
            "records_with": recs_hit,
            "pct": round(100.0 * recs_hit / n, 1) if n else 0.0,
            "occurrences": occ_n,
        }
        occ[topic] = occ_n

        if topic in NOTABLE_TOPICS and hit_records:
            for r in hit_records[:12]:
                notable.append({
                    "openalex_id": r.get("openalex_id"),
                    "title": r.get("title"),
                    "year": r.get("year"),
                    "authors": r.get("authors") or [],
                    "matched_topic": topic,
                    "is_oa": bool(r.get("is_oa")),
                    "oa_url": r.get("oa_url") or "",
                    "doi": r.get("doi") or "",
                })

    processed = {
        "source": "OpenAlex via scripts/fetch_external_200plus.py",
        "total_records": n,
        "year_min": y_min,
        "year_max": y_max,
        "with_abstract": abs_n,
        "open_access_urls": oa_n,
        "abstract_coverage_pct": round(100.0 * abs_n / n, 1) if n else 0.0,
        "oa_pct": round(100.0 * oa_n / n, 1) if n else 0.0,
        "counts": counts,
        "occurrences": occ,
        "notable": notable,
        "caveats": [
            "Keyword matching is abstract-level only.",
            "Findings are labeled abstract-only unless OA full text was retrieved.",
            "Unmentioned dimensions are recorded as not-stated-in-abstract.",
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    md = ["# External 200+ Papers — Processed Analysis", "",
          f"- Source: OpenAlex (via `scripts/fetch_external_200plus.py`).",
          f"- Total records: **{n}** (years {y_min}–{y_max}).",
          f"- With abstract: **{abs_n}** ({processed['abstract_coverage_pct']}%) · Open access URLs: **{oa_n}** ({processed['oa_pct']}%).",
          "",
          "## Exact keyword / regex counts across the 200 abstracts",
          "",
          "| topic | records_with | pct_of_records | occurrences |",
          "|---|---|---|---|"]
    for topic, c in counts.items():
        md.append(f"| {topic} | {c['records_with']} | {c['pct']:.1f}% | {c['occurrences']} |")
    md += [
        "",
        "## Caveats / labeling",
        "",
        "All findings above are **abstract-only** unless an OA full text was retrieved; dimensions not matched are `not-stated-in-abstract`.",
        "",
        "## Notable papers (explicitly discussing multi-horizon, returns/leakage, or news/GDELT sentiment)",
        "",
        "| openalex_id | title | year | matched_topic | is_oa |",
        "|---|---|---|---|---|",
    ]
    for e in notable:
        md.append(f"| {e['openalex_id']} | {e['title']} | {e['year']} | {e['matched_topic']} | {e['is_oa']} |")
    md.append("")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps({
        "total_records": n,
        "year_min": y_min,
        "year_max": y_max,
        "with_abstract": abs_n,
        "open_access_urls": oa_n,
        "counts": counts,
        "notable_count": len(notable),
        "json_out": args.out_json,
        "md_out": args.out_md,
    }, indent=2))


if __name__ == "__main__":
    main()

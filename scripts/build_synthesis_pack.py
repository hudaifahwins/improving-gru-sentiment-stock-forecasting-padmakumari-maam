#!/usr/bin/env python3
"""Build a stratified reading pack (markdown) from literature_corpus/.

Grounds the synthesis in actually-read text: all 18 faculty-paper seed
references (full abstracts), every paper whose abstract discusses leakage /
look-ahead / overlapping windows / shuffled splits (full abstracts), a sample
of good-practice papers (walk-forward, train-only scaling, baselines), the
top-cited paper of each 50-paper batch, and the overall top-cited papers.

Output: literature_corpus/_reading_pack.md (gitignored intermediate).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "literature_corpus")
OUT = os.path.join(CORPUS, "_reading_pack.md")
BATCH_SIZE = 50


def parse_md(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    fm, abstract = {}, ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            try:
                fm = json.loads(text[3:end])
            except Exception:
                fm = {}
            rest = text[end + 4:]
            m = re.search(r"## Abstract\s*\n(.*?)(?=\n## |\Z)", rest, re.S)
            if m:
                abstract = m.group(1).strip()
    return fm, abstract


def trunc(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def block(rec, fm, abstract, full=True, show_file=True):
    lim = None if full else 480
    a = abstract if full else trunc(abstract, 480)
    lines = [
        f"### {fm.get('title') or 'Untitled'}",
        f"- **year:** {fm.get('year')} · **venue:** {fm.get('venue') or 'n/a'} · "
        f"**cited:** {fm.get('cited_by', 0)} · **OA:** {fm.get('is_oa')} · "
        f"**seed:** {fm.get('seed')}",
    ]
    if show_file:
        lines.append(f"- **file:** `literature_corpus/{rec['file']}`")
    if fm.get("doi"):
        lines.append(f"- **doi:** {fm['doi']}")
    lines += ["", a if a else "_no abstract_", ""]
    return "\n".join(lines)


def main():
    records = []
    with open(os.path.join(CORPUS, "_extraction.jsonl"), encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    by_file = {r["file"]: r for r in records}
    shown = set()
    out = ["# Literature reading pack (grounding for docs/literature_synthesis.md)", ""]

    # --- Section 1: seed references (faculty paper's own bibliography) ---
    out += ["## 1. Seed — the 18 references of Padmakumari et al. (2026)", ""]
    for r in records:
        if r["seed"]:
            fm, abstract = parse_md(os.path.join(CORPUS, r["file"]))
            out.append(block(r, fm, abstract, full=True))
            shown.add(r["openalex_id"])

    # --- Section 2: leakage / gap papers (full abstracts) ---
    gap_keys = ("Look-ahead/leakage discussed", "Overlapping windows", "Random/shuffled split")
    gap_recs = [r for r in records if any(g in r["gaps"] for g in gap_keys)]
    out += [f"## 2. Papers whose abstracts discuss leakage / look-ahead / overlap / shuffled splits ({len(gap_recs)})", ""]
    for r in gap_recs:
        fm, abstract = parse_md(os.path.join(CORPUS, r["file"]))
        out.append(block(r, fm, abstract, full=True))
        out.append(f"- **detected gaps:** {', '.join(r['gaps'])}")
        out.append("")
        shown.add(r["openalex_id"])

    # --- Section 3: good-practice sample (walk-forward / train-only scaling / baseline) ---
    practice_recs = [
        r for r in records
        if {"Walk-forward/OOS", "Train-only scaling", "Baseline included"} & set(r["practices"])
    ]
    practice_recs.sort(key=lambda r: r["cited_by"] or 0, reverse=True)
    out += [f"## 3. Good-practice sample — walk-forward / train-only scaling / baselines (top {min(12, len(practice_recs))} by citations of {len(practice_recs)})", ""]
    for r in practice_recs[:12]:
        if r["openalex_id"] in shown:
            continue
        fm, abstract = parse_md(os.path.join(CORPUS, r["file"]))
        out.append(block(r, fm, abstract, full=False))
        out.append(f"- **practices:** {', '.join(r['practices'])}")
        out.append("")
        shown.add(r["openalex_id"])

    # --- Section 4: top-cited per batch ---
    out += ["## 4. Top-cited paper of each 50-paper batch", ""]
    for b in range(0, len(records), BATCH_SIZE):
        chunk = records[b : b + BATCH_SIZE]
        if not chunk:
            continue
        top = max(chunk, key=lambda r: r["cited_by"] or 0)
        if top["openalex_id"] in shown:
            continue
        fm, abstract = parse_md(os.path.join(CORPUS, top["file"]))
        out.append(f"#### Batch {b // BATCH_SIZE + 1:02d} top-cited")
        out.append(block(top, fm, abstract, full=False))
        shown.add(top["openalex_id"])

    # --- Section 5: overall top-cited ---
    out += ["## 5. Overall most-cited papers in corpus", ""]
    topcited = sorted(records, key=lambda r: r["cited_by"] or 0, reverse=True)
    n = 0
    for r in topcited:
        if r["openalex_id"] in shown:
            continue
        fm, abstract = parse_md(os.path.join(CORPUS, r["file"]))
        out.append(block(r, fm, abstract, full=False))
        shown.add(r["openalex_id"])
        n += 1
        if n >= 15:
            break

    # --- Section 6: scaling-mention papers (titles only, for scaler-leakage mapping) ---
    scaling = [r for r in records if "Scaling/normalization mentioned" in r["gaps"]]
    out += [f"## 6. Papers mentioning scaling/normalization ({len(scaling)}) — titles only", ""]
    for r in scaling:
        out.append(f"- {r['year']} · {r['title']} — {r['venue'] or 'n/a'} (cited {r['cited_by'] or 0}) [`{r['file']}`]")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {OUT}  ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()

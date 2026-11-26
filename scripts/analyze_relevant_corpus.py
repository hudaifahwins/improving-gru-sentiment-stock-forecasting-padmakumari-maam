#!/usr/bin/env python3
"""Gap-oriented audit of the method-driven corpus expansion.

Reads `literature_corpus/relevant_2026-11-26/_manifest.json` (written by
`scripts/fetch_relevant_papers.py`) and reports, for every project gap, how
many abstracts in the expansion mention it. Every count is computed from the
manifest's verbatim OpenAlex abstracts/titles — nothing is generated.

Outputs:
  <corpus>/ANALYSIS.json          machine-readable counts + exemplar records
  <corpus>/ANALYSIS_TABLE.md      the markdown table used in the docs memo

Usage:
    python3 scripts/analyze_relevant_corpus.py
    python3 scripts/analyze_relevant_corpus.py --corpus literature_corpus/relevant_2026-11-26
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Signal name -> regex over "title + abstract". Patterns are deliberately
# narrow enough to require the concept, not just a stray word.
SIGNALS: dict[str, str] = {
    # targets
    "target_price_level": r"stock price|closing price|price level|price prediction|price forecasting|predict(?:ing|ion of)? (?:the )?(?:stock |closing )?price",
    "target_return": r"stock returns?|return prediction|predict(?:ing|ion of)? returns?|excess returns?|abnormal returns?|cross[- ]section(?:al)? returns?",
    "target_direction": r"direction(?:al)? (?:prediction|accuracy|classification)|market direction|price movement|trend prediction|up[/-]down",
    "target_volatility": r"volatility (?:forecast|prediction|model)|realized volatility|\bgarch\b",
    "input_volume": r"trading volume|\bvolume\b|order book|order flow",
    # evaluation rigor
    "walk_forward": r"walk[- ]forward|rolling window|rolling origin|expanding window|time[- ]series cross[- ]validation|rolling cross[- ]validation",
    "non_overlapping": r"non[- ]?overlap|non[- ]overlapping|disjoint (?:window|sample|test|evaluation)",
    "significance_test": r"diebold[- ]mariano|statistical(?:ly)? significan|significance test|p[- ]values?|hypothesis test|bootstrap(?:ped)? (?:confidence|test|interval)",
    "multiple_testing": r"multiple (?:testing|comparisons)|holm[- ]?(?:bonferroni)?|benjamini|false discovery|data snooping|family[- ]wise",
    "train_only_scaling": r"train(?:ing)?[- ]only|fit(?:ted)? (?:the )?scaler (?:on|using) (?:the )?train|scaler.{0,40}train",
    "transaction_costs": r"transaction costs?|trading costs?|trading fees?|slippage|commissions?",
    "backtest": r"backtest|trading strategy|portfolio simulation|sharpe",
    "leakage": r"look[- ]ahead|data leakage|information leakage|\bleakage\b",
    # sentiment
    "sent_finbert": r"finbert",
    "sent_llm": r"large language models?|\bllms?\b|chatgpt|gpt-?[345]",
    "sent_lexicon": r"vader|loughran|lm dictionary|sentiment lexicon|textblob|sentiwordnet",
    "sent_gdelt": r"\bgdelt\b",
    "sent_social": r"twitter|twitter/x|reddit|stocktwits|social media",
    "sent_news": r"news (?:sentiment|headline|article|media)|financial news|news[- ]based",
    "sent_filings": r"10-?k|8-?k|sec filing|annual report|earnings call",
    # markets
    "market_korea": r"\bkorea|korean|kospi|kosdaq|samsung|005930",
    "market_us": r"\bu\.?s\.? market|\bu\.?s\.? stock|nasdaq|nyse|s&p ?500|dow jones|wall street",
    "market_china": r"\bchina|chinese|shanghai|shenzhen|csi ?300|a-?shares?",
    "market_india": r"\bindia|indian|nifty|\bbse\b|sensex",
    # architectures / baselines
    "arch_gru": r"\bgru\b|gated recurrent unit",
    "arch_lstm": r"\blstm\b|long short[- ]term memory",
    "arch_transformer": r"transformer",
    "arch_gnn": r"graph neural|\bgcn\b|\bgat\b|graph convolution",
    "arch_ensemble": r"ensemble|stacking|boosting|xgboost|lightgbm|random forest",
    "arch_rl": r"reinforcement learning|\bdqn\b|\bppo\b|q[- ]learning",
    "baseline_persistence": r"persistence|naive (?:forecast|baseline|prediction)|random walk",
}

# Counts from the audited 575-paper citation-lineage corpus
# (docs/literature_synthesis.md). Used only as a comparison column; None means
# the lineage synthesis does not report a matching figure.
LINEAGE_575: dict[str, int | None] = {
    "target_price_level": 278,   # 48.3%
    "target_return": 82,         # 14.3%
    "target_volatility": 17,     # 3.0%
    "input_volume": 12,          # volume/order-book bucket, 2.1%
    "non_overlapping": 0,
    "significance_test": 20,     # 3.5%
    "train_only_scaling": 0,
    "sent_gdelt": 0,
    "market_korea": 2,           # 0.3%
    "arch_gru": 75,              # 13.0%
    "arch_lstm": 254,            # 44.2%
    "arch_transformer": 63,
    "arch_gnn": 35,
    "sent_finbert": 53,          # BERT/FinBERT bucket
}

# Which exemplars to surface per gap (most cited first).
EXEMPLAR_SIGNALS = [
    "target_return", "target_direction", "non_overlapping", "significance_test",
    "multiple_testing", "transaction_costs", "train_only_scaling", "leakage",
    "sent_finbert", "sent_gdelt", "sent_llm", "market_korea",
]


def hit(regex: str, text: str) -> bool:
    return re.search(regex, text, re.I) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="literature_corpus/relevant_2026-11-26")
    args = ap.parse_args()
    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "_manifest.json").read_text(encoding="utf-8"))
    papers = manifest["papers"]
    n = len(papers)

    counts: Counter[str] = Counter()
    by_theme: dict[str, Counter[str]] = {}
    exemplars: dict[str, list[dict]] = {}
    for rec in papers:
        text = f"{rec.get('title', '')} {rec.get('abstract', '')}"
        theme = rec.get("theme") or "unknown"
        by_theme.setdefault(theme, Counter())
        for name, regex in SIGNALS.items():
            if hit(regex, text):
                counts[name] += 1
                by_theme[theme][name] += 1
        for name in EXEMPLAR_SIGNALS:
            if hit(SIGNALS[name], text):
                exemplars.setdefault(name, []).append({
                    "title": rec.get("title"),
                    "year": rec.get("year"),
                    "venue": rec.get("venue"),
                    "cited_by": rec.get("cited_by_count") or 0,
                    "doi": rec.get("doi"),
                    "oa_url": rec.get("oa_url") or rec.get("oa_pdf_url"),
                    "openalex_id": rec.get("openalex_id"),
                })

    for name in exemplars:
        exemplars[name].sort(key=lambda r: (-r["cited_by"], -(r["year"] or 0)))

    out = {
        "corpus": str(corpus),
        "n": n,
        "with_abstract": sum(1 for r in papers if r.get("abstract")),
        "open_access": sum(1 for r in papers if r.get("is_oa")),
        "signals": dict(counts),
        "by_theme": {t: dict(c) for t, c in by_theme.items()},
        "exemplars": {k: v[:5] for k, v in exemplars.items()},
    }
    (corpus / "ANALYSIS.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    def pct(k: int) -> str:
        return f"{100.0 * k / n:.1f}%"

    lines = [
        "| signal | papers | share | lineage 575 | delta (pp) |",
        "|---|---|---|---|---|",
    ]
    for name in SIGNALS:
        k = counts[name]
        lin = LINEAGE_575.get(name)
        if lin is None:
            lin_s, delta = "—", "—"
        else:
            lin_s = f"{lin} ({100.0 * lin / 575:.1f}%)"
            delta = f"{100.0 * k / n - 100.0 * lin / 575:+.1f}"
        lines.append(f"| `{name}` | {k} | {pct(k)} | {lin_s} | {delta} |")
    lines += ["", "| theme | papers |", "|---|---|"]
    theme_counts = Counter(r.get("theme") or "unknown" for r in papers)
    for t, c in theme_counts.most_common():
        lines.append(f"| {t} | {c} |")
    (corpus / "ANALYSIS_TABLE.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")

    print(f"[n] {n} papers · abstract {out['with_abstract']} · OA {out['open_access']}")
    for name in SIGNALS:
        print(f"{name:24s} {counts[name]:5d} {pct(counts[name]):>7s}")
    print(f"\nwrote {corpus / 'ANALYSIS.json'}\nwrote {corpus / 'ANALYSIS_TABLE.md'}")


if __name__ == "__main__":
    main()

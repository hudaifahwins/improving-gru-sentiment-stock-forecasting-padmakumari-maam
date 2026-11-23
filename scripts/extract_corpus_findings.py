#!/usr/bin/env python3
"""Batch extraction over literature_corpus/*.md — reads EVERY paper file.

Parses the YAML-frontmatter JSON (written by fetch_literature_md.py) plus the
abstract, then detects architecture / modality / target / horizon / metric /
good-practice / gap signals via keyword regexes. Output:

  literature_corpus/_extraction.jsonl   one record per paper (all signals)
  literature_corpus/_batches/batch_XX.md  per-batch notes (~50 papers each)
  literature_corpus/_corpus_stats.json  corpus-wide aggregates

Everything is grounded in the actual file contents. No numbers are invented:
only signal presence/absence per paper is recorded.
"""
import collections
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "literature_corpus")
BATCH_DIR = os.path.join(CORPUS, "_batches")
BATCH_SIZE = 50

SIGNALS = {
    "arch": [
        ("GRU", r"\bgru\b|gated recurrent unit"),
        ("LSTM", r"\blstm\b|long short[- ]term memory"),
        ("BiGRU", r"bi[- ]?gru|bidirectional gru"),
        ("BiLSTM", r"bi[- ]?lstm|bidirectional lstm"),
        ("CNN", r"\bcnn\b|convolutional neural"),
        ("CNN-LSTM/GRU", r"cnn[- ]?(lstm|gru)|convolutional.{0,25}(lstm|gru|recurrent)"),
        ("Transformer", r"transformer"),
        ("BERT/FinBERT", r"\bbert\b|finbert|roberta|electra|language model"),
        ("Attention mechanism", r"attention mechanism|self[- ]attention|attention layer|attention module"),
        ("Vanilla RNN", r"\brnn\b|recurrent neural network"),
        ("MLP/feedforward", r"multilayer perceptron|\bmlp\b|feedforward"),
        ("Autoencoder", r"auto[- ]?encoder"),
        ("VAE", r"variational autoencoder|\bvae\b"),
        ("GAN", r"generative adversarial|\bgan\b"),
        ("TCN", r"temporal convolutional|\btcn\b"),
        ("TFT", r"temporal fusion transformer|\btft\b"),
        ("Modern time-series Transformer", r"n[- ]beats|deepar|informer|autoformer|patchtst|timesnet|fedformer|\bformer\b"),
        ("XGBoost", r"xgboost"),
        ("LightGBM", r"lightgbm"),
        ("Random forest", r"random forest"),
        ("SVM/SVR", r"support vector|\bsvm\b|\bsvr\b"),
        ("Ridge/linear", r"\bridge\b|linear regression|elastic net|\blasso\b|least squares"),
        ("ARIMA/SARIMA", r"\barima\b|sarima"),
        ("GARCH", r"\bgarch\b"),
        ("Prophet", r"\bprophet\b"),
        ("Persistence/naive", r"persistence|naive (forecast|baseline|prediction)|random walk"),
        ("Momentum/technical rule", r"momentum|moving average crossover|trading rule|technical trading rule"),
        ("Ensemble/stacking", r"ensemble|stacking|bagging|boosting"),
        ("Reinforcement learning", r"reinforcement learning|\bdqn\b|\bppo\b|q[- ]learning|deep q"),
        ("Graph NN", r"graph neural|\bgcn\b|\bgat\b|graph convolution"),
        ("LSTM-GRU hybrid", r"hybrid.{0,80}(lstm|gru)|(lstm|gru).{0,80}hybrid|combined.{0,80}(lstm|gru)|(lstm|gru).{0,80}combined"),
        ("Echo state/reservoir", r"echo state|reservoir computing"),
        ("kNN", r"\bknn\b|k[- ]nearest neighbor"),
    ],
    "modalities": [
        ("OHLCV prices", r"ohlc|open.{0,3}high.{0,3}low.{0,3}close|closing price|close price|price series|historical price|daily price"),
        ("Technical indicators", r"technical indicator|\brsi\b|\bmacd\b|bollinger|moving average|stochastic|williams|\badx\b|\bcci\b|\batr\b"),
        ("Realized/implied volatility", r"realized volatility|implied volatility|\bvix\b|volatility index"),
        ("Volume/order book", r"trading volume|order book|limit order|\btick\b|level[- ]?2"),
        ("News articles", r"news (articles?|headlines?)|financial news|news sentiment|news flow"),
        ("Twitter/social", r"twitter|\btweet|social media|reddit|stocktwits|weibo|online forum"),
        ("GDELT", r"gdelt"),
        ("Search trends", r"google trends|search volume|search engine|baidu index"),
        ("Macro", r"macroeconomic|interest rate|inflation|\bcpi\b|\bgdp\b|fomc|monetary policy|federal funds"),
        ("Fundamentals", r"fundamental|earnings|price[- ]to[- ]earnings|\bp/e\b|financial statement|\b10[- ]k\b|sec filing|balance sheet"),
        ("Crypto", r"bitcoin|cryptocurrenc|\bethereum\b"),
        ("Forex", r"foreign exchange|\bforex\b|exchange rate|currency pair"),
        ("Commodities", r"\bgold\b|crude oil|\boil\b|commodity|silver"),
        ("Indices", r"s&p ?500|\bnifty\b|\bsensex\b|nikkei|\bdax\b|\bftse\b|hang seng|\bkospi\b|shanghai composite|stock index|market index"),
        ("Options/derivatives", r"\boptions?\b|futures contract|derivative"),
    ],
    "targets": [
        ("Price level", r"price (forecast|prediction|level|movement)|closing price|close price|future price|price trend"),
        ("Returns", r"return(s| rate)?\b|log return|percentage change|excess return|\byield\b"),
        ("Direction (classification)", r"directional|direction of|up or down|rise or fall|classification|binary|up[- ]down|movement direction|sign of"),
        ("Volatility", r"volatility (forecast|prediction)|variance|realized vol"),
    ],
    "horizons": [
        ("Intraday/high-frequency", r"intraday|minute|high[- ]frequency"),
        ("Next-day (h=1)", r"next[- ]day|one[- ]day|1[- ]day|daily forecast|daily prediction|h ?= ?1|one.day ahead"),
        ("Multi-day (h=3/5+)", r"multi[- ]step|multi[- ]horizon|3[- ]day|5[- ]day|three.day|five.day|weekly forecast|one.week ahead|h ?= ?[357]"),
        ("Monthly/long-term", r"monthly|long.term|yearly|annual"),
    ],
    "metrics": [
        ("RMSE", r"rmse|root mean squar"),
        ("MAE", r"\bmae\b|mean absolute error"),
        ("MAPE", r"mape|mean absolute percentage"),
        ("R2", r"r.?squared|\br2\b|coefficient of determination"),
        ("Directional accuracy", r"directional accuracy|hit ratio|direction accuracy|direction.of"),
        ("Classification acc", r"\baccuracy\b|\bf1\b|\bprecision\b|\brecall\b|\bauc\b|confusion matrix"),
        ("Returns/Sharpe", r"sharpe|cumulative return|\bprofit\b|trading return|strategy return|annualized return"),
        ("Statistical tests", r"diebold|mariano|\bdm\b test|model confidence set|\bmcs\b|p[- ]?value|statistically significant|\bt[- ]?test|wilcoxon|paired test"),
        ("Other metrics", r"theil|smape|wmape|rmsle|mean absolute scaled|\bmas\b"),
    ],
    "practices": [
        ("Walk-forward/OOS", r"walk[- ]forward|rolling window|expanding window|out[- ]of[- ]sample|rolling.origin"),
        ("Train-only scaling", r"fit on (the )?train|training (set|data) only|train.only|scaler.{0,40}train"),
        ("Baseline included", r"baseline|benchmark|naive|persistence|random walk|buy.and.hold|historical average"),
        ("Significance tested", r"diebold|mariano|\bdm\b test|model confidence set|\bmcs\b|p[- ]?value|statistically significant|\bt[- ]?test|wilcoxon|paired test|significance test"),
    ],
    "gaps": [
        ("Overlapping windows", r"overlap|overlapping (window|sample|observation|period)"),
        ("Scaling/normalization mentioned", r"normaliz|standardiz|min[- ]max|minmax|scaler|z.score"),
        ("Look-ahead/leakage discussed", r"look[- ]ahead|leakage|data leak|future information|peeking"),
        ("Random/shuffled split", r"random (split|shuffle|partition)|shuffled|randomly (split|assign)"),
        ("Single ticker/market", r"single (stock|ticker|market|company|asset)|one (stock|ticker|company|market)|individual stock|a single"),
        ("China/A-share", r"\bchina\b|a[- ]share|shanghai|shenzhen|chinese"),
        ("India", r"\bindia\b|nifty|sensex|\bbse\b|\bnse\b|indian"),
        ("US market", r"s&p ?500|nasdaq|\bnyse\b|u\.?s\.? market|american|wall street"),
        ("Korea", r"korea|kospi|korean"),
        ("Emerging markets", r"emerging market|developing (country|market)"),
        ("Crypto market", r"crypto|digital asset"),
    ],
}


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


def detect(haystack, patterns):
    return [label for label, rx in patterns if re.search(rx, haystack, re.I)]


def main():
    os.makedirs(BATCH_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CORPUS, "*.md")))
    records = []
    for path in files:
        if os.path.basename(path).startswith("_"):
            continue  # skip intermediates (_reading_pack.md) — papers only
        fm, abstract = parse_md(path)
        hay = " ".join([
            str(fm.get("title", "")), abstract,
            str(fm.get("venue", "")), " ".join(fm.get("topics", []) or []),
        ])
        rec = {
            "file": os.path.basename(path),
            "openalex_id": fm.get("openalex_id"),
            "title": fm.get("title"),
            "year": fm.get("year"),
            "venue": fm.get("venue"),
            "doi": fm.get("doi"),
            "cited_by": fm.get("cited_by", 0),
            "is_oa": fm.get("is_oa", False),
            "seed": fm.get("seed", False),
            "abstract_words": len(abstract.split()) if (abstract and "No abstract available" not in abstract) else 0,
        }
        for cat in SIGNALS:
            rec[cat] = detect(hay, SIGNALS[cat])
        records.append(rec)

    with open(os.path.join(CORPUS, "_extraction.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Corpus-wide stats
    def counter(cat):
        c = collections.Counter()
        for r in records:
            c.update(r[cat])
        return c

    years = sorted({r["year"] for r in records if r["year"]})
    venues = collections.Counter(r["venue"] or "unknown" for r in records)
    stats = {
        "total_papers": len(records),
        "year_range": [years[0], years[-1]] if years else None,
        "year_histogram": dict(sorted(collections.Counter(r["year"] for r in records if r["year"]).items())),
        "with_abstract": sum(1 for r in records if r["abstract_words"] > 0),
        "open_access": sum(1 for r in records if r["is_oa"]),
        "seed_papers": sum(1 for r in records if r["seed"]),
        "top_venues": venues.most_common(15),
        "top_cited": sorted(
            [{"title": r["title"], "year": r["year"], "venue": r["venue"], "cited_by": r["cited_by"], "file": r["file"]}
             for r in records], key=lambda x: x["cited_by"], reverse=True)[:20],
        "signal_counts": {cat: dict(counter(cat).most_common()) for cat in SIGNALS},
        "papers_with_any_gap": sum(1 for r in records if r["gaps"]),
        "papers_with_practices": sum(1 for r in records if r["practices"]),
    }
    with open(os.path.join(CORPUS, "_corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Batch notes
    def trunc(s, n=70):
        s = str(s or "").replace("|", "/").replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    for b in range(0, len(records), BATCH_SIZE):
        chunk = records[b : b + BATCH_SIZE]
        idx0, idx1 = b + 1, b + len(chunk)
        lines = [
            f"# Batch {b // BATCH_SIZE + 1:02d} — papers {idx0}–{idx1} of {len(records)}",
            "",
            f"Generated from literature_corpus/ by scripts/extract_corpus_findings.py (signal detection over title+abstract+venue+topics).",
            "",
            "## Papers",
            "",
            "| # | file | yr | venue | cited | title | architectures | modalities | targets | horizons | metrics | practices | gaps |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(chunk, start=idx0):
            cells = [
                r["file"], str(r["year"] or ""), trunc(r["venue"], 24), str(r["cited_by"] or 0),
                trunc(r["title"]),
                ", ".join(r["arch"]) or "—", ", ".join(r["modalities"]) or "—",
                ", ".join(r["targets"]) or "—", ", ".join(r["horizons"]) or "—",
                ", ".join(r["metrics"]) or "—", ", ".join(r["practices"]) or "—",
                ", ".join(r["gaps"]) or "—",
            ]
            lines.append(f"| {i} | " + " | ".join(cells) + " |")
        lines += ["", "## Batch signal summary", ""]
        for cat in SIGNALS:
            c = collections.Counter()
            for r in chunk:
                c.update(r[cat])
            top = ", ".join(f"{k} ×{v}" for k, v in c.most_common(12))
            lines.append(f"- **{cat}**: {top or '—'}")
        with open(os.path.join(BATCH_DIR, f"batch_{b // BATCH_SIZE + 1:02d}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "papers_processed": len(records),
        "batches": (len(records) + BATCH_SIZE - 1) // BATCH_SIZE,
        "year_range": stats["year_range"],
        "with_abstract": stats["with_abstract"],
        "open_access": stats["open_access"],
        "seed_papers": stats["seed_papers"],
    }, indent=2))


if __name__ == "__main__":
    main()

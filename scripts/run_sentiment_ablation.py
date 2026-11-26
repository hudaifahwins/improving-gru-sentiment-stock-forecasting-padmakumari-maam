#!/usr/bin/env python3
"""Does sentiment add anything under the honest protocol?

Paired ablation on the two tickers for which a daily news-sentiment series
exists (AAPL, 005930.KS). Everything else is identical to
``run_decisive_multiday.py``: expanding walk-forward origins, disjoint
evaluation windows (stride = h), h-day log-return target, train-only scaling,
3 GRU seeds averaged, 5 bps/side, Diebold-Mariano (HAC).

Arms per origin:
  hist_mean      - expanding historical mean of the h-day return
  ridge_price    - Ridge on the 6 price/volatility features
  ridge_sent     - Ridge on price features + sentiment block
  gru_price      - GRU on price features only (the anchor design minus text)
  gru_sent       - GRU on price features + sentiment block (anchor design)

Two pre-registered test families, Holm-corrected separately:
  A) each arm vs hist_mean            (does the model beat the mean?)
  B) gru_sent vs gru_price            (does sentiment add to the GRU?)
     and ridge_sent vs ridge_price

Also reports the raw predictive content of sentiment: Pearson / Spearman
correlation of sentiment_t with the next h-day return, and a McNemar exact
test on the discordant direction calls of gru_sent vs gru_price.

Outputs: results/sentiment_ablation/summary.json + findings.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_decisive_multiday as base  # noqa: E402
from src.vol_gated_gru import hac_dm  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "results" / "sentiment_ablation"
LOOKBACK = base.LOOKBACK
COST = base.COST


def load_sentiment_frame(ticker: str) -> pd.DataFrame:
    path = ROOT / "data" / f"prices_sentiment_{ticker}.csv"
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def price_features(df: pd.DataFrame) -> np.ndarray:
    """Same six features as the main harness (close/vol/momentum block)."""
    return base.features(df)


def sentiment_block(s: np.ndarray) -> np.ndarray:
    """Sentiment level and its 3-day mean (causal, no look-ahead)."""
    ser = pd.Series(s)
    mean3 = ser.rolling(3, min_periods=1).mean().to_numpy()
    return np.column_stack([s, mean3]).astype(np.float32)


def holm(pvals: list[float]) -> list[float]:
    return base.holm(pvals)


def sentiment_ic(df: pd.DataFrame, horizons: list[int]) -> list[dict]:
    """Unconditional association between sentiment and future returns."""
    from scipy.stats import pearsonr, spearmanr

    close = df["close"].to_numpy(float)
    s = df["sentiment"].to_numpy(float)
    out = []
    for h in horizons:
        fwd = np.full(len(close), np.nan)
        fwd[:-h] = np.log(close[h:] / close[:-h]) if h < len(close) else np.nan
        m = np.isfinite(fwd)
        pr = pearsonr(s[m], fwd[m])
        sr = spearmanr(s[m], fwd[m])
        out.append({
            "horizon": h, "n": int(m.sum()),
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "note": "overlapping windows; p-values assume iid (upper bound on "
                    "significance), effective n ~ n/h",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,005930.KS")
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--origins", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    tickers = args.tickers.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    torch.set_num_threads(1)

    cells: list[dict] = []
    ics: list[dict] = []
    paired: list[dict] = []

    for ticker in tickers:
        df = load_sentiment_frame(ticker)
        close = df["close"].to_numpy(float)
        s = df["sentiment"].to_numpy(float)
        f_price = price_features(df)
        f_sent = np.column_stack([f_price, sentiment_block(s)]).astype(np.float32)
        n = len(close)
        print(f"# {ticker}: n={n} {df.date.min().date()}..{df.date.max().date()}",
              flush=True)
        for ic in sentiment_ic(df, horizons):
            ics.append({"ticker": ticker, **ic})
        for h in horizons:
            origins = base.origins_for(n, h, args.origins)
            print(f"{ticker} h={h} n_origins={len(origins)}", flush=True)
            series: dict[str, dict] = {}

            def record(key, p, y):
                d = series.setdefault(key, {"p": [], "y": [], "spread": []})
                d["p"].append(p)
                d["y"].append(y)

            for org in origins:
                scaled = {}
                for tag, feat in (("price", f_price), ("sent", f_sent)):
                    mu = feat[:org].mean(0)
                    sd = feat[:org].std(0) + 1e-8
                    scaled[tag] = (feat - mu) / sd
                x_tr_p, y_tr = base.make_xy(scaled["price"], close, org, h)
                # Same training windows, but with the sentiment block appended.
                t_idx = np.arange(LOOKBACK - 1, org - h)
                x_tr_s = np.stack([scaled["sent"][i - LOOKBACK + 1: i + 1]
                                   for i in t_idx])
                x_te_p = scaled["price"][org - LOOKBACK: org]
                x_te_s = scaled["sent"][org - LOOKBACK: org]
                y = float(np.log(close[org - 1 + h] / close[org - 1]))

                record("hist_mean", float(np.mean(y_tr)), y)
                record("ridge_price", base.fit_predict(
                    "ridge", x_tr_p, y_tr, x_te_p, h, 42, 0), y)
                record("ridge_sent", base.fit_predict(
                    "ridge", x_tr_s, y_tr, x_te_s, h, 42, 0), y)
                for key, xtr, xte in (("gru_price", x_tr_p, x_te_p),
                                      ("gru_sent", x_tr_s, x_te_s)):
                    ps = [base.fit_predict("gru", xtr, y_tr, xte, h, sd, args.epochs)
                          for sd in seeds]
                    record(key, float(np.mean(ps)), y)
                    series[key]["spread"].append(float(np.std(ps)))

            for model, d in series.items():
                y = np.array(d["y"]); p = np.array(d["p"])
                act, prd = np.sign(y), np.sign(p)
                dir_acc = float(np.mean(act == prd))
                majority = max(float(np.mean(act > 0)), float(np.mean(act < 0)))
                hm = np.array(series["hist_mean"]["p"])
                _, dm_p_hm = hac_dm(np.abs(y - p), np.abs(y - hm))
                r = np.expm1(y)
                flips = float(np.sum(np.abs(np.diff(np.concatenate([[0], prd])))))
                net = float(np.sum(prd * r)) - COST * flips
                cells.append({
                    "ticker": ticker, "horizon": h, "model": model,
                    "n": int(len(y)), "dir_acc": dir_acc,
                    "majority_base": majority, "excess": dir_acc - majority,
                    "dm_p_vs_hist_mean": float(dm_p_hm),
                    "net_return_5bps": net, "turnover": flips / len(y),
                    "seed_spread": float(np.mean(d["spread"])) if d["spread"] else None,
                })

            for sent_key, price_key in (("gru_sent", "gru_price"),
                                        ("ridge_sent", "ridge_price")):
                ds = series[sent_key]; dp = series[price_key]
                y = np.array(ds["y"])
                ps = np.array(ds["p"]); pp = np.array(dp["p"])
                _, dm_p = hac_dm(np.abs(y - ps), np.abs(y - pp))
                paired.append({
                    "ticker": ticker, "horizon": h,
                    "comparison": f"{sent_key} vs {price_key}",
                    "n": int(len(y)),
                    "dir_acc_sent": float(np.mean(np.sign(y) == np.sign(ps))),
                    "dir_acc_price": float(np.mean(np.sign(y) == np.sign(pp))),
                    "delta_dir_acc": float(np.mean(np.sign(y) == np.sign(ps))
                                           - np.mean(np.sign(y) == np.sign(pp))),
                    "mse_sent": float(np.mean((y - ps) ** 2)),
                    "mse_price": float(np.mean((y - pp) ** 2)),
                    "dm_p_sent_vs_price": float(dm_p),
                    "mcnemar_p": None,
                })
                # McNemar on correctness of direction calls
                cs = np.sign(y) == np.sign(ps)
                cp = np.sign(y) == np.sign(pp)
                n01 = int(np.sum(cs & ~cp)); n10 = int(np.sum(~cs & cp))
                from scipy.stats import binomtest
                paired[-1]["mcnemar_p"] = (
                    float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
                    if n01 + n10 else 1.0)
                paired[-1]["discordant"] = [n01, n10]

    # Family A: vs hist_mean. Family B: sent vs price.
    a_ps = [c["dm_p_vs_hist_mean"] for c in cells]
    for c, a in zip(cells, holm(a_ps)):
        c["p_holm_vs_hist_mean"] = float(a)
    b_ps = [p["dm_p_sent_vs_price"] for p in paired]
    for p, a in zip(paired, holm(b_ps)):
        p["p_holm_sent_vs_price"] = float(a)

    payload = {
        "protocol": {
            "lookback": LOOKBACK, "non_overlapping": True,
            "cost_bps_per_side": 5.0, "seeds": seeds,
            "origins_cap": args.origins, "epochs": args.epochs,
            "sentiment_features": ["level", "3-day mean"],
            "target": "h-day log return",
        },
        "sentiment_ic": ics, "cells": cells, "paired": paired,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    L = ["# Sentiment ablation — does news sentiment add value?", "",
         f"- paired design on the two sentiment-covered tickers; up to "
         f"{args.origins} disjoint origins per cell; 5 bps/side; "
         f"seeds {seeds}; GRU 15 epochs", "",
         "## Unconditional sentiment content", "",
         "| ticker | h | n | Pearson r | p | Spearman r | p |",
         "|---|---|---|---|---|---|---|"]
    for ic in ics:
        L.append(f"| {ic['ticker']} | {ic['horizon']} | {ic['n']} | "
                 f"{ic['pearson_r']:+.4f} | {ic['pearson_p']:.4g} | "
                 f"{ic['spearman_r']:+.4f} | {ic['spearman_p']:.4g} |")
    L += ["", "## Arm-level results", "",
          "| ticker | h | model | n | DirAcc | majority | excess | DM p | Holm | net5bps | seed sd |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        sd = f"{c['seed_spread']:.3f}" if c["seed_spread"] is not None else "-"
        L.append(f"| {c['ticker']} | {c['horizon']} | {c['model']} | {c['n']} | "
                 f"{c['dir_acc']:.4f} | {c['majority_base']:.4f} | "
                 f"{c['excess']:+.4f} | {c['dm_p_vs_hist_mean']:.4g} | "
                 f"{c['p_holm_vs_hist_mean']:.4g} | {c['net_return_5bps']:+.4f} | {sd} |")
    L += ["", "## Paired test: sentiment vs price-only", "",
          "| ticker | h | comparison | n | DirAcc sent | DirAcc price | delta | MSE sent | MSE price | DM p | Holm | McNemar p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in paired:
        L.append(f"| {p['ticker']} | {p['horizon']} | {p['comparison']} | {p['n']} | "
                 f"{p['dir_acc_sent']:.4f} | {p['dir_acc_price']:.4f} | "
                 f"{p['delta_dir_acc']:+.4f} | {p['mse_sent']:.6f} | "
                 f"{p['mse_price']:.6f} | {p['dm_p_sent_vs_price']:.4g} | "
                 f"{p['p_holm_sent_vs_price']:.4g} | {p['mcnemar_p']:.4g} |")
    (out / "findings.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'findings.md'} ({len(cells)} cells, {len(paired)} paired)")


if __name__ == "__main__":
    main()

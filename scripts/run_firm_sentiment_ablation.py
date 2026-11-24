"""Firm-level sentiment ablation on AAPL (real SEC EDGAR 8-K + LM lexicon).

Three sentiment settings on identical AAPL splits (h=3/5, 70/15/15 + expanding
WF, seeds 42/123/2026):

  - no_sent  : price+tech+ohlcv_volnorm only (no sentiment feature)
  - gdelt    : cached GDELT tone (aggregate news tone, lagged)
  - firm      : local firm-level LM-lexicon polarity from real SEC 8-K text
                (data/sentiment_aapl_firm.csv, strictly lagged 1 day)

005930.KS remains on GDELT (no SEC 8-K coverage) — reported as reference only.

Metrics per cell: reconstructed MAE/RMSE/nRMSE/MAPE, DirAcc (full OOS + 3/3
seed consensus), excess over majority-sign base, k/3 collapse, Newey-West/HAC
DM stat & p-value (vs base and vs no_sent), turnover, 5bps/side net return.

Outputs: results/firm_sentiment_ablation/summary.json (gitignored) + tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_consensus_conviction as rcc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "firm_sentiment_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIRM = Path("data/sentiment_aapl_firm.csv")
GDELT = Path("data/sentiment_AAPL.csv")


def make_sentiment_df(ticker: str, source: str) -> pd.DataFrame:
    df = rcc.load_ohlcv(ticker)
    if ticker == "AAPL" and source == "firm" and FIRM.exists():
        firm = pd.read_csv(FIRM)
        firm["Date"] = pd.to_datetime(firm["Date"]).dt.normalize()
        firm = firm.sort_values("Date").set_index("Date").reindex(df["Date"]).ffill()
        df = df.copy()
        df["Sentiment"] = firm["polarity"].values
        df["NewsCount"] = firm["count"].fillna(0).values
    elif ticker == "AAPL" and source == "gdelt" and GDELT.exists():
        sent = pd.read_csv(GDELT)
        sent["Date"] = pd.to_datetime(sent["Date"]).dt.normalize()
        df = df.merge(sent[["Date", "Sentiment"]], on="Date", how="left")
        df["NewsCount"] = 0.0
    else:
        if "Sentiment" not in df.columns:
            df["Sentiment"] = np.nan
        df["NewsCount"] = 0.0
    return df


def evaluate_row(mask: np.ndarray, y: np.ndarray, p: np.ndarray,
                 baseline_err: np.ndarray | None) -> dict[str, Any]:
    return rcc.evaluate(mask, y, p, baseline_err)


def run_cell(ticker: str, source: str, horizon: int, split: str,
             seed: int, epochs: int) -> dict[str, Any]:
    df = make_sentiment_df(ticker, source)
    model = "no_sent_mse" if source == "none" else "concat_sent_gru"
    cfg = rcc.Cfg(ticker, horizon, seed, model, split, epochs)
    out = rcc._run_split(cfg, df, wf=(split == "wf"))
    y, p = out["y"], out["p"]
    prev = out["prev"]
    e = (y - p) ** 2
    e_base = y ** 2
    full = evaluate_row(np.ones(len(y), bool), y, p, e_base)
    q = out["extra"]["q_val"]
    thr = np.quantile(np.abs(p), q)
    conv = evaluate_row(np.abs(p) > thr, y, p, e_base)
    from src.baseline import compute_metrics
    p_hat = prev * np.exp(p)
    price_metrics = compute_metrics(prev * np.exp(y), p_hat, prev)
    return {
        "ticker": ticker, "source": source, "horizon": horizon,
        "split": split, "seed": seed, "model": model,
        "full": full, "conviction": conv,
        "n_test_total": out["n_tot"], "price_metrics": price_metrics,
        "y": y.tolist(), "p": p.tolist(), "prev": prev.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rows: list[dict[str, Any]] = []
    for split in ("701515", "wf"):
        for source in ("none", "gdelt", "firm"):
            for horizon in (3, 5):
                per_seed = []
                for seed in seeds:
                    t0 = time.perf_counter()
                    per_seed.append(run_cell("AAPL", source, horizon, split,
                                             seed, args.epochs))
                    print(f"AAPL {split} {source} h={horizon} seed={seed} "
                          f"({time.perf_counter()-t0:.1f}s)", flush=True)
                # 3/3 consensus using stored per-seed predictions
                n = min(len(s["y"]) for s in per_seed)
                y0 = np.array(per_seed[0]["y"][:n])
                preds = np.array([s["p"][:n] for s in per_seed])
                agree = np.all(np.sign(preds) == np.sign(preds[0]), axis=0)
                cons = evaluate_row(agree, y0, preds[0], y0 ** 2)
                rows.append({
                    "ticker": "AAPL", "split": split, "source": source,
                    "horizon": horizon, "seeds": seeds, "per_seed": per_seed,
                    "consensus_3of3": cons,
                })
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"summary saved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

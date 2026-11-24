#!/usr/bin/env python3
"""Focused leak-free follow-up: multi-horizon return/direction perspective ablation.

For tickers {AAPL, 005930.KS} and horizons {3, 5}, compares GRU configurations on
strictly chronological 70/15/15 splits with causal windows:

  - no-sentiment baseline: price+tech+ohlcv_volnorm (no sentiment features)
  - gated sentiment:       news-activity-gated lagged GDELT sentiment
  - unconditioned sentiment: lagged GDELT tone/count features (no gate)

All runs reuse the verified harness modules (src.baseline / src.horizon_experiment /
src.return_experiment / src.features_research) so data loading, leakage controls,
splitting, scaling, windowing, and training match the verified battery.

Usage: python3 -m scripts.run_perspective_ablation --tickers AAPL,005930.KS --horizons 3,5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import (  # noqa: E402
    DataConfig,
    ModelConfig,
    resolve_data_source,
    resolve_sentiment_columns,
)
from src.horizon_experiment import HConfig, run_one  # noqa: E402


def dir_acc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(y)))


def gate_sentiment(df, sent_cols) -> object:
    """Gate lagged sentiment by rolling 10-day active-news count (>=1 -> keep)."""
    if not sent_cols:
        return df
    sent_col = sent_cols[0]
    tone = df[sent_col].astype(float).to_numpy(dtype=float)
    active = (df[sent_col].astype(float) != 0).to_numpy(dtype=float)
    act = pd.Series(active).rolling(10, min_periods=1).sum().to_numpy(dtype=float)
    gate = (act >= 1).astype(float)
    df2 = df.copy()
    df2[sent_col] = pd.Series(tone * gate, index=df.index, dtype=float)
    return df2


def run_one_cfg(ticker: str, *, horizon: int, features: str, stride: str, loss: str,
               seed: int, out_dir, df=None, gated: bool = False) -> dict:
    if df is None:
        df, _src = resolve_data_source(path=None, ticker=ticker, synthetic=False,
                                      seed=seed, sentiment_cols=())
    sent_cols: tuple = ()
    if "sentiment" in features:
        s = resolve_sentiment_columns(df)
        sent_cols = tuple(c.strip() for c in s if c.strip()) if s else ()
    data_cfg = DataConfig(sentiment_cols=sent_cols)
    if gated and sent_cols:
        df = gate_sentiment(df, sent_cols)
    cfg = HConfig(
        horizon=horizon, loss=loss, baseline="none", deadzone_eps=0.005,
        features=features, stride=stride, vol_deadzone=0.0, seed=seed,
        out_dir=out_dir,
    )
    model_cfg = ModelConfig(
        seed=seed, model="gru", hidden_size=50, num_layers=1, epochs=200,
        lr=1e-3, patience=15, batch_size=64, device="auto",
    )
    payload = run_one(df, data_cfg=data_cfg, cfg=cfg, model_cfg=model_cfg,
                      source_label=f"perspective_ablation:{ticker}")
    r = payload["results"]["gru"]
    rm = r["return_metrics"]
    w = payload["results"]["windows"]
    y = np.array(w["y"], dtype=float)
    pred = np.array(w["pred"], dtype=float)
    mom = np.array([np.nan if v is None else float(v) for v in w["momentum_sign"]], dtype=float)
    valid = ~np.isnan(mom)
    base_up = float((y > 0).mean())
    base_down = float((y < 0).mean())
    base_maj = max(base_up, base_down)
    d_acc = float(np.mean(np.sign(pred) == np.sign(y)))
    mom_acc = float(np.mean(mom[valid] == np.sign(y[valid]))) if valid.sum() > 0 else float("nan")
    ridge_res = payload["results"].get("ridge") or {}
    return {
        "ticker": ticker,
        "horizon": horizon,
        "features": features,
        "stride": stride,
        "loss": loss,
        "dir_acc": float(rm["directional_accuracy"]),
        "dir_acc_strict": float(rm["directional_accuracy_strict"]),
        "excess_over_base": float(d_acc - base_maj),
        "base_majority_up": base_up,
        "base_majority_down": base_down,
        "base_majority": float(base_maj),
        "n_windows": int(len(y)),
        "mom_dir_acc": float(mom_acc),
        "mom_n_valid": int(valid.sum()),
        "collapsed": bool(len(np.unique(np.sign(pred))) == 1),
        "ridge_dir_acc": (float(ridge_res["return_metrics"]["directional_accuracy"])
                          if ridge_res else None),
        "gru_price_mae": float(r["price_metrics"]["mae"]),
        "gru_price_rmse": float(r["price_metrics"]["rmse"]),
        "epochs_run": int(r["epochs_run"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,005930.KS")
    ap.add_argument("--horizons", default="3,5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/perspective_ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    for t in tickers:
        for h in horizons:
            for feat, gated, label in (
                ("price+tech+ohlcv_volnorm", False, "no_sent"),
                ("price+tech+ohlcv_volnorm+sentiment", True, "gated"),
                ("price+tech+ohlcv_volnorm+sentiment", False, "uncond"),
            ):
                df, src = resolve_data_source(path=None, ticker=t, synthetic=False,
                                             seed=args.seed, sentiment_cols=())
                res = run_one_cfg(
                    t, horizon=h, features=feat, stride="nonoverlap", loss="mse",
                    seed=args.seed, out_dir=out_dir, df=df, gated=gated,
                )
                res["variant"] = label
                results.append(res)
                print(json.dumps(res))

    summary = {
        "source": "scripts/run_perspective_ablation.py",
        "tickers": tickers,
        "horizons": horizons,
        "stride": "nonoverlap",
        "loss": "mse",
        "seed": args.seed,
        "runs": results,
        "note": "Gated sentiment gates lagged tone by rolling 10-day active-news count (>=1).",
    }
    with open(out_dir / "perspective_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"runs": len(results),
                     "out": str(out_dir / "perspective_ablation_summary.json")}))


if __name__ == "__main__":
    main()

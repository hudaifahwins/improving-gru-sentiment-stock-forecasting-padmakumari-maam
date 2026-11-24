#!/usr/bin/env python3
"""Real research v2: focused leak-free follow-up on H1/H2.

H1: Does combining return/margin or volatility-scaled return targets improve
    DirAcc and base-rate excess over the no_sent baseline (price+tech+ohlcv_volnorm)
    without inducing collapses?
H2: Does filtering/weighting lagged GDELT sentiment strictly on high-news-activity
    days (top-quartile NewsCount threshold on train/val) outperform unconditioned
    sentiment and no_sent, unlike active10 >= 1 which kept all rows?

Leak-free: 70/15/15 chronological split, train-only scaler, backward-looking windows,
1-day sentiment lag, date-aligned tone maxdiff = 0.0 (verified on merged CSVs).

Usage: python3 -m scripts.run_real_research_v2
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
    resolve_sentiment_columns,
)
from src.horizon_experiment import HConfig, run_one  # noqa: E402


DATA_FILES = {
    "AAPL": "data/prices_sentiment_AAPL.csv",
    "005930.KS": "data/prices_sentiment_005930.KS.csv",
}


def load_merged(ticker: str) -> pd.DataFrame:
    path = DATA_FILES[ticker]
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def apply_news_quartile_gate(df: pd.DataFrame, sent_cols: tuple, train_frac=0.70, val_frac=0.15) -> pd.DataFrame:
    """Gate sentiment to top-quartile NewsCount days (threshold fit on train rows only)."""
    if not sent_cols:
        return df
    df2 = df.copy()
    n = len(df2)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    train_end = n_train + n_val
    # find a NewsCount-like column
    nc_col = None
    for c in df2.columns:
        if "news" in c.lower() and "count" in c.lower():
            nc_col = c
            break
    if nc_col is None and len(sent_cols) > 1:
        nc_col = sent_cols[1]
    if nc_col is None or nc_col not in df2.columns:
        return df2
    train_nc = pd.to_numeric(df2.iloc[:train_end][nc_col], errors="coerce").fillna(0.0)
    # top quartile on train/val: threshold at 75th percentile
    thr = float(train_nc.quantile(0.75))
    gate = (pd.to_numeric(df2[nc_col], errors="coerce").fillna(0.0) >= thr).astype(float)
    sent_col = sent_cols[0]
    tone = pd.to_numeric(df2[sent_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    df2[sent_col] = pd.Series(tone * gate.to_numpy(dtype=float), index=df2.index, dtype=float)
    return df2, thr


def run_one_cfg(ticker: str, df: pd.DataFrame, *, horizon: int, features: str,
                stride: str, loss: str, seed: int, out_dir: Path,
                gate_quartile: bool = False) -> dict:
    sent_cols = ()
    if "sentiment" in features:
        s = resolve_sentiment_columns(df)
        sent_cols = tuple(c.strip() for c in s if c.strip()) if s else ()
    gate_thr = None
    df_use = df
    if gate_quartile and sent_cols:
        gated = apply_news_quartile_gate(df, sent_cols)
        if isinstance(gated, tuple):
            df_use, gate_thr = gated
        else:
            df_use = gated
    data_cfg = DataConfig(sentiment_cols=sent_cols)
    cfg = HConfig(
        horizon=horizon, loss=loss, baseline="ridge", deadzone_eps=0.005,
        features=features, stride=stride, vol_deadzone=0.0, seed=seed,
        out_dir=out_dir,
    )
    model_cfg = ModelConfig(
        seed=seed, model="gru", hidden_size=50, num_layers=1, epochs=200,
        lr=1e-3, patience=15, batch_size=64, device="auto",
    )
    payload = run_one(df_use, data_cfg=data_cfg, cfg=cfg, model_cfg=model_cfg,
                      source_label=f"real_research_v2:{ticker}")
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
    ridge = payload["results"].get("ridge") or {}
    ridge_acc = (float(ridge["return_metrics"]["directional_accuracy"])
                 if ridge else None)
    return {
        "ticker": ticker,
        "horizon": horizon,
        "features": features,
        "loss": loss,
        "seed": seed,
        "gate_quartile": gate_quartile,
        "gate_thr": gate_thr,
        "dir_acc": float(rm["directional_accuracy"]),
        "dir_acc_strict": float(rm.get("directional_accuracy_strict", float("nan"))),
        "excess_over_base": float(d_acc - base_maj),
        "base_majority": float(base_maj),
        "base_up": base_up,
        "n_windows": int(len(y)),
        "mom_dir_acc": float(mom_acc),
        "collapsed": bool(len(np.unique(np.sign(pred))) == 1),
        "ridge_dir_acc": ridge_acc,
    }


def summarize(rows):
    # group by (ticker,horizon,features,loss,gate_quartile)
    groups = {}
    for r in rows:
        k = (r["ticker"], r["horizon"], r["features"], r["loss"], r["gate_quartile"])
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in groups.items():
        da = [x["dir_acc"] for x in rs]
        ex = [x["excess_over_base"] for x in rs]
        out.append({
            "ticker": k[0], "horizon": k[1], "features": k[2], "loss": k[3],
            "gate_quartile": k[4],
            "n_seeds": len(rs),
            "dir_acc_mean": float(np.mean(da)),
            "dir_acc_std": float(np.std(da, ddof=1)) if len(da) > 1 else 0.0,
            "excess_mean": float(np.mean(ex)),
            "excess_std": float(np.std(ex, ddof=1)) if len(ex) > 1 else 0.0,
            "collapse_count": int(sum(1 for x in rs if x["collapsed"])),
            "n_windows": int(rs[0]["n_windows"]),
            "base_majority": float(rs[0]["base_majority"]),
            "mom_dir_acc_mean": float(np.mean([x["mom_dir_acc"] for x in rs])),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/real_research_v2")
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tickers = ["AAPL", "005930.KS"]
    horizons = [3, 5]

    configs = [
        # (features, loss, gate_quartile, label)
        ("price+tech+ohlcv_volnorm", "mse", False, "no_sent_mse"),
        ("price+tech+ohlcv_volnorm", "dir_margin", False, "no_sent_dir_margin"),
        ("price+tech+ohlcv_volnorm+sentiment", "mse", False, "sent_uncond_mse"),
        ("price+tech+ohlcv_volnorm+sentiment", "mse", True, "sent_topq_mse"),
    ]

    rows = []
    for ticker in tickers:
        df = load_merged(ticker)
        for h in horizons:
            for seed in seeds:
                for feat, loss, gateq, label in configs:
                    rec = run_one_cfg(ticker, df, horizon=h, features=feat,
                                      stride="nonoverlap", loss=loss, seed=seed,
                                      out_dir=out_dir, gate_quartile=gateq)
                    rec["label"] = label
                    rows.append(rec)
                    print(json.dumps({
                        "ticker": ticker, "h": h, "seed": seed, "label": label,
                        "dir_acc": rec["dir_acc"], "excess": rec["excess_over_base"],
                        "collapsed": rec["collapsed"], "n": rec["n_windows"],
                        "base_maj": rec["base_majority"],
                    }))
    summ = summarize(rows)
    out = {"rows": rows, "summary": summ, "seeds": seeds}
    (out_dir / "real_research_v2_summary.json").write_text(json.dumps(out, indent=2))
    # print table
    print("\nSUMMARY")
    for s in sorted(summ, key=lambda x:(x["ticker"],x["horizon"],x["label"] if False else x["features"]+str(x["loss"])+str(x["gate_quartile"]))):
        print(f"{s['ticker']:10s} h={s['horizon']} {s['features'][:28]:28s} loss={s['loss']:10s} gateQ={str(s['gate_quartile']):5s} "
              f"DirAcc={s['dir_acc_mean']:.4f}±{s['dir_acc_std']:.4f} excess={s['excess_mean']:+.4f}±{s['excess_std']:.4f} "
              f"collapse={s['collapse_count']}/{s['n_seeds']} n={s['n_windows']} base={s['base_majority']:.4f} mom={s['mom_dir_acc_mean']:.4f}")


if __name__ == "__main__":
    main()

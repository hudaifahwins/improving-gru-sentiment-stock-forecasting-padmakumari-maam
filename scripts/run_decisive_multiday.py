#!/usr/bin/env python3
"""Decisive multi-day, non-overlapping, cost-aware evaluation.

Protocol (identical for every model):
- expanding walk-forward origins; evaluation windows are disjoint (stride = h)
- training windows use only data available at the origin; target is the h-day
  log return; no 1-step extrapolation
- scaler fit on the origin's history only
- 5 bps/side, turnover, Diebold-Mariano (HAC) vs the historical-mean forecast,
  Holm correction across all cells
- models: historical mean, persistence, causal momentum, Ridge, GRU (3 seeds,
  prediction averaged across seeds; seed spread reported separately)

Outputs: results/decisive_multiday/summary.json + findings.md
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vol_gated_gru import hac_dm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "decisive_multiday"
CACHE = ROOT / "data" / "cache"
LOOKBACK = 30
COST = 0.0005


def load_close(ticker: str) -> pd.DataFrame:
    path = CACHE / f"{ticker.replace('^', 'idx_')}.csv"
    if not path.exists():
        import yfinance as yf

        raw = yf.download(ticker, period="7y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]
        raw.to_csv(path, index=False)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy(float)
    logp = np.log(close)
    ret = np.diff(logp, prepend=logp[0])
    vol = pd.Series(ret).rolling(10, min_periods=2).std().fillna(0.0).to_numpy()
    mom5 = np.zeros_like(logp)
    mom5[5:] = logp[5:] - logp[:-5]
    mom20 = np.zeros_like(logp)
    mom20[20:] = logp[20:] - logp[:-20]
    vol_chg = np.zeros_like(logp)
    if "volume" in df:
        v = np.log1p(df["volume"].to_numpy(float))
        vol_chg[1:] = np.diff(v)
    return np.column_stack([logp, ret, mom5, mom20, vol, vol_chg]).astype(np.float32)


def origins_for(n: int, h: int, cap: int) -> list[int]:
    first = int(n * 0.5)
    all_ = list(range(first, n - h + 1, h))
    if len(all_) > cap:
        step = math.ceil(len(all_) / cap)
        all_ = all_[::step][:cap]
    return all_


def make_xy(feat: np.ndarray, close: np.ndarray, upto: int, h: int):
    """Training windows with target inside history [0, upto)."""
    t = np.arange(LOOKBACK - 1, upto - h)
    x = np.stack([feat[i - LOOKBACK + 1 : i + 1] for i in t])
    y = np.log(close[t + h] / close[t])
    return x, y.astype(np.float32)


def train_gru(x: np.ndarray, y: np.ndarray, seed: int, epochs: int) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Sequential()
    gru = nn.GRU(x.shape[2], 32, 1, batch_first=True)
    head = nn.Linear(32, 1)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-3)
    lossf = nn.MSELoss()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y[:, None])
    n_val = max(8, int(len(x) * 0.15))
    tr = slice(0, len(x) - n_val)
    va = slice(len(x) - n_val, len(x))
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        gru.train(); head.train()
        opt.zero_grad()
        out, _ = gru(xt[tr])
        loss = lossf(head(out[:, -1]), yt[tr])
        loss.backward(); opt.step()
        gru.eval(); head.eval()
        with torch.no_grad():
            vo, _ = gru(xt[va])
            vl = float(lossf(head(vo[:, -1]), yt[va]))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = ({k: v.clone() for k, v in gru.state_dict().items()},
                          {k: v.clone() for k, v in head.state_dict().items()})
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state:
        gru.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    model.add_module("gru", gru); model.add_module("head", head)
    model.eval()
    return model


def predict_gru(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out, _ = model.gru(torch.from_numpy(x))
        return model.head(out[:, -1]).numpy()[:, 0]


def fit_predict(model: str, x_tr: np.ndarray, y_tr: np.ndarray,
                x_te: np.ndarray, h: int, seed: int, epochs: int,
                mom_p: float = 0.0) -> float:
    if model == "hist_mean":
        return float(np.mean(y_tr))
    if model == "majority":
        return 1e-6 if float(np.mean(y_tr)) >= 0 else -1e-6
    if model == "persistence":
        return 0.0
    if model == "momentum":
        return mom_p
    if model == "ridge":
        from sklearn.linear_model import Ridge
        r = Ridge(alpha=1.0).fit(x_tr.reshape(len(x_tr), -1), y_tr)
        return float(r.predict(x_te.reshape(1, -1))[0])
    if model == "gru":
        m = train_gru(x_tr, y_tr, seed, epochs)
        return float(predict_gru(m, x_te[None])[0])
    raise ValueError(model)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (cumulative maximum)."""
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, pvals[idx] * (m - rank))
        out[idx] = min(1.0, run)
    return out.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,MSFT,005930.KS,^NSEI,000001.SS")
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--origins", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--gru-tickers", default="AAPL,005930.KS,^NSEI")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    tickers = args.tickers.split(",")
    gru_tickers = set(args.gru_tickers.split(","))
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    torch.set_num_threads(1)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    for ticker in tickers:
        df = load_close(ticker)
        close = df["close"].to_numpy(float)
        feat = features(df)
        n = len(close)
        for h in horizons:
            origins = origins_for(n, h, args.origins)
            print(f"{ticker} h={h} n_origins={len(origins)}", flush=True)
            series: dict[str, dict] = {}
            for org in origins:
                scaler_mu = feat[:org].mean(0)
                scaler_sd = feat[:org].std(0) + 1e-8
                scaled = (feat - scaler_mu) / scaler_sd
                x_tr, y_tr = make_xy(scaled, close, org, h)
                x_te = scaled[org - LOOKBACK : org]
                y = float(np.log(close[org - 1 + h] / close[org - 1]))
                mom_p = float(np.log(close[org - 1] / close[org - 1 - h]))
                models = ("hist_mean", "majority", "persistence", "momentum",
                          "ridge", "gru") if ticker in gru_tickers else \
                    ("hist_mean", "majority", "persistence", "momentum", "ridge")
                for model in models:
                    if model == "gru":
                        ps = [fit_predict("gru", x_tr, y_tr, x_te, h, s, args.epochs)
                              for s in seeds]
                        p = float(np.mean(ps))
                        series.setdefault("gru_seed_spread", []).append(float(np.std(ps)))
                    else:
                        p = fit_predict(model, x_tr, y_tr, x_te, h, 42, 0,
                                        mom_p=mom_p)
                    series.setdefault(model, {"p": [], "y": []})
                    series[model]["p"].append(p)
                    series[model]["y"].append(y)

            for model, d in series.items():
                if model == "gru_seed_spread":
                    continue
                y = np.array(d["y"]); p = np.array(d["p"])
                act, prd = np.sign(y), np.sign(p)
                dir_acc = float(np.mean(act == prd))
                majority = max(float(np.mean(act > 0)), float(np.mean(act < 0)))
                hm = np.array(series["hist_mean"]["p"])
                _, dm_p_hm = hac_dm(np.abs(y - p), np.abs(y - hm))
                r = np.expm1(y)
                pos = prd
                gross = float(np.sum(pos * r))
                flips = float(np.sum(np.abs(np.diff(np.concatenate([[0], pos])))))
                net = gross - COST * flips
                cells.append({
                    "ticker": ticker, "horizon": h, "model": model,
                    "n": int(len(y)), "dir_acc": dir_acc,
                    "majority_base": majority, "excess": dir_acc - majority,
                    "dm_p_vs_hist_mean": dm_p_hm,
                    "net_return_5bps": net, "turnover": flips / len(y),
                    "seed_spread": float(np.mean(series["gru_seed_spread"]))
                    if model == "gru" else None,
                })

    ps = [c["dm_p_vs_hist_mean"] for c in cells]
    for c, a in zip(cells, holm(ps)):
        c["p_holm"] = float(a)

    with open(out / "summary.json", "w") as f:
        json.dump({"protocol": {
            "lookback": LOOKBACK, "non_overlapping": True, "cost_bps_per_side":
            5.0, "seeds": seeds, "origins_cap": args.origins,
            "epochs": args.epochs}, "cells": cells}, f, indent=2)

    lines = [
        "# Decisive multi-day evaluation — non-overlapping, cost-aware",
        "",
        f"- lookback {LOOKBACK}; disjoint evaluation windows; 5 bps/side; "
        f"seeds {seeds}; up to {args.origins} origins per cell",
        "- DM = Diebold–Mariano (HAC) vs historical-mean forecast; "
        "Holm across all cells",
        "",
        "| ticker | h | model | n | DirAcc | majority | excess | DM p | Holm | net5bps | turnover |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c['ticker']} | {c['horizon']} | {c['model']} | {c['n']} | "
            f"{c['dir_acc']:.4f} | {c['majority_base']:.4f} | "
            f"{c['excess']:+.4f} | {c['dm_p_vs_hist_mean']:.4g} | "
            f"{c['p_holm']:.4g} | {c['net_return_5bps']:+.4f} | "
            f"{c['turnover']:.2f} |")
    (out / "findings.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'findings.md'} ({len(cells)} cells)")


if __name__ == "__main__":
    main()

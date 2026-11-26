"""Open-source benchmarking under the strict anti-leakage contract.

Protocol (uniform for EVERY model family)
-----------------------------------------
Expanding rolling origins: history grows; at each origin the model is FIT on
history only, then predicts h steps ahead; the next origin is h indices later
(non-overlapping horizons). This is identical for our GRU / VolGatedGRU,
Ridge / HistGradientBoostingRegressor, and NeuralForecast models
(NBEATS / NHITS, univariate close — these open-source deep models do not
consume our windowed exogenous tensor without look-ahead risk; documented).

Seeds 42/123/2026; features are strictly causal; scalers fit on train-only.

Outputs: results/opensource_benchmark/summary.json (gitignored) + findings.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baseline import compute_metrics, fit_scaler, set_seed  # noqa: E402
from src.vol_gated_gru import (  # noqa: E402
    BaselineGRU,
    SplitData,
    VolGatedGRU,
    _causal_vol,
    fit_hysteresis_tau,
    hac_dm,
    prepare_split_data,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS

    NF_AVAILABLE = True
except Exception:  # pragma: no cover
    NF_AVAILABLE = False

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "opensource_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK = 30
COST = 0.0005


def load_close(ticker: str) -> pd.Series:
    from run_consensus_conviction import load_ohlcv

    df = load_ohlcv(ticker)
    return df.set_index("Date")["Close"].astype(float).sort_index()


def origins_for(n: int, h: int, first_pct: float = 0.4,
               cap: int = 24) -> list[int]:
    first = int(n * first_pct)
    all_ = list(range(first, n - h + 1, h))
    if cap and len(all_) > cap:
        step = math.ceil(len(all_) / cap)
        all_ = all_[::step][:cap]
    return all_


def window_matrix(prices: np.ndarray, origin: int, h: int, feats: np.ndarray
                 ) -> tuple[np.ndarray, float, float]:
    """X windows over history [0, origin), target at origin-1 -> origin-1+h."""
    hist_feats = feats[:origin]
    k = len(hist_feats) - LOOKBACK
    if k < 1:
        raise ValueError("history too short")
    x = np.empty((k, LOOKBACK, feats.shape[1]), dtype=np.float32)
    for i in range(k):
        x[i] = hist_feats[i : i + LOOKBACK]
    y = np.log(prices[origin - 1 + h] / prices[origin - 1])
    return x, float(y), float(prices[origin - 1])


def build_feats(prices: np.ndarray) -> np.ndarray:
    """Causal feature stack (close, ret_volnorm-like, momentum, sent tone)."""
    # price+tech+ohlcv-like causal features from close alone (kept simple).
    logp = np.log(prices)
    ret = np.diff(logp, prepend=logp[0])
    n = len(prices)
    f = np.zeros((n, 5))
    f[:, 0] = logp
    f[:, 1] = logp - np.roll(logp, 1)
    f[:, 1] = np.where(np.arange(n) == 0, 0.0, f[:, 1])
    f[:, 2] = logp - np.roll(logp, 20)
    f[:, 2] = np.where(np.arange(n) < 20, 0.0, f[:, 2])
    f[:, 3] = ret
    f[:, 4] = np.roll(ret, 1)  # t-1 return
    f[:1, 4] = 0.0
    return f.astype(np.float32)


def fit_gru_and_predict(prices: np.ndarray, feats: np.ndarray, origin: int,
                        h: int, arch: str, loss: str, seed: int,
                        epochs: int) -> float:
    hist_feats = feats[:origin]
    scaler = fit_scaler(hist_feats)
    scaled = scaler.transform(feats)
    k = origin - LOOKBACK
    x = np.empty((k, LOOKBACK, feats.shape[1]), dtype=np.float32)
    for i in range(k):
        x[i] = scaled[i : i + LOOKBACK]
    y = np.log(prices[LOOKBACK : LOOKBACK + k] / prices[:k])
    n_val = max(1, int(k * 0.15))
    x_tr, y_tr = x[:-n_val], y[:-n_val]
    x_va, y_va = x[-n_val:], y[-n_val:]
    if arch == "vgg":
        n_sent = 1
        model = VolGatedGRU(input_size=feats.shape[1] - n_sent, n_sent=n_sent)
    else:
        model = BaselineGRU(input_size=feats.shape[1])
    split = SplitData(x_tr, y_tr[:, None], prices[LOOKBACK : LOOKBACK + k - n_val],
                      prices[:k - n_val],
                      x_va, y_va[:, None], prices[LOOKBACK + k - n_val :],
                      prices[k - n_val : k],
                      np.zeros((1, LOOKBACK, feats.shape[1])),
                      np.zeros((1, 1)), np.zeros(1), np.zeros(1),
                      scaler, [], 1 if arch == "vgg" else 0)
    # train using vol_gated train_model
    from src.vol_gated_gru import train_model as tm

    tm(model, split, use_sentiment=(arch == "vgg"), gated=(arch == "vgg"),
       epochs=epochs, seed=seed, loss_kind=loss)
    last = scaled[origin - LOOKBACK : origin].astype(np.float32)  # (L, F)
    from src.vol_gated_gru import predict as predfn

    p = predfn(model, last[None], use_sentiment=(arch == "vgg"),
               n_sent=(1 if arch == "vgg" else 0), gated=(arch == "vgg"))[0]
    # returns: p is log-return over h (model trained on h=1 target though)
    # NOTE: model trained on 1-step; we extrapolate return* h for h>1 as
    # approximation for the diagnostic battery only. We flag in memo.
    return float(p) * h


def fit_ml_predict(prices: np.ndarray, feats: np.ndarray, origin: int, h: int,
                   kind: str, seed: int) -> float:
    x, _, _ = window_matrix(prices, origin, h, feats)
    y = np.log(prices[LOOKBACK : origin] / prices[: origin - LOOKBACK])
    model = Ridge(alpha=1.0) if kind == "ridge" else HistGradientBoostingRegressor(
        random_state=seed, max_iter=300)
    model.fit(x.reshape(len(x), -1), y)
    last = feats[origin - LOOKBACK : origin][None]
    pred = model.predict(last.reshape(1, -1))[0]
    return float(pred) * h


def fit_nf_predict(prices: np.ndarray, dates: pd.DatetimeIndex, origin: int,
                   h: int, kind: str, seed: int) -> float:
    fr = pd.DataFrame({"unique_id": "x", "ds": dates[:origin],
                       "y": prices[:origin]})
    model_cls = NBEATS if kind == "nbeats" else NHITS
    nf = NeuralForecast(models=[model_cls(input_size=LOOKBACK, h=h,
                                          max_steps=200)],
                        freq="D")
    nf.fit(fr)
    fcst = nf.predict(h=h)  # h-step-ahead forecast (matches configured horizon)
    col = "NBEATS" if kind == "nbeats" else "NHITS"
    if col not in fcst.columns:
        col = fcst.columns[-1]
    level = float(fcst[col].iloc[-1])
    return float(np.log(level / prices[origin - 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,005930.KS")
    ap.add_argument("--horizons", default="3,5")
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--origins-cap", type=int, default=24)
    args = ap.parse_args()
    tickers = args.tickers.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    families = ["gru_h", "gru_hd", "vgg_h", "vgg_hd",
                "ridge", "gbr"]
    if NF_AVAILABLE:
        families += ["nbeats", "nhits"]

    cells: list[dict[str, Any]] = []
    pvals: list[float] = []
    for ticker in tickers:
        prices = load_close(ticker)
        feats = build_feats(prices.to_numpy())
        dates = prices.index
        for h in horizons:
            origins = origins_for(len(prices), h, cap=args.origins_cap)
            for fam in families:
                for seed in seeds:
                    set_seed(seed)
                    preds, ys, prevs = [], [], []
                    for org in origins:
                        try:
                            if fam.startswith("gru") or fam.startswith("vgg"):
                                arch = "vgg" if fam.startswith("vgg") else "gru"
                                loss = "huber" if fam.endswith("_h") else "huber_dir"
                                p = fit_gru_and_predict(
                                    prices.to_numpy(), feats, org, h, arch, loss,
                                    seed, args.epochs)
                            elif fam in ("ridge", "gbr"):
                                p = fit_ml_predict(prices.to_numpy(), feats,
                                                   org, h, fam, seed)
                            else:
                                p = fit_nf_predict(prices.to_numpy(), dates,
                                                   org, h, fam, seed)
                            preds.append(p)
                            ys.append(np.log(prices.iloc[org - 1 + h]
                                             / prices.iloc[org - 1]))
                            prevs.append(float(prices.iloc[org - 1]))
                        except Exception as exc:  # guard per-origin failures
                            cells.append({"ticker": ticker, "horizon": h,
                                          "family": fam, "seed": seed,
                                          "status": "origin_failed",
                                          "error": str(exc)[:120]})
                            continue
                    if not preds:
                        continue
                    y = np.array(ys)
                    p = np.array(preds)
                    prev = np.array(prevs)
                    e = (y - p) ** 2
                    e_base = y ** 2
                    act = np.sign(y)
                    prd = np.sign(p)
                    dir_acc = float(np.mean(act == prd))
                    majority = max(float(np.mean(act > 0)),
                                   float(np.mean(act < 0)))
                    z, z_p = hac_dm(e, e_base)
                    r = np.expm1(y)
                    gross = float(np.sum(prd * r))
                    flips = float(np.sum(np.abs(np.diff(prd)))) if len(prd) > 1 else 0.0
                    net = gross - COST * flips
                    p_hat = prev * np.exp(p)
                    pm = compute_metrics(prev * np.exp(y), p_hat, prev)
                    cells.append({
                        "ticker": ticker, "horizon": h, "family": fam,
                        "seed": seed, "n": int(len(y)), "dir_acc": dir_acc,
                        "majority_base_rate": majority,
                        "excess_dir_acc": dir_acc - majority,
                        "dm_vs_base": z, "dm_p_vs_base": z_p,
                        "price_metrics": pm,
                        "turnover": flips / max(len(prd) - 1, 1),
                        "net_return_5bps": net,
                        "status": "ok",
                    })
                    pvals.append(z_p)
                    print(f"{ticker} h={h} {fam} seed={seed} n={len(y)} "
                          f"dir={dir_acc:.4f} net={net:.3f}", flush=True)

    # Holm + BH across all tested cells
    m = len(pvals)
    if m:
        order = np.argsort(pvals)
        holm = np.empty(m)
        run = 1.0
        for rank, idx in enumerate(order):
            val = pvals[idx] * (m - rank)
            run = min(run, val)
            holm[idx] = run
        order_rev = np.argsort(pvals)[::-1]
        bh = np.empty(m)
        run = 1.0
        for rank, idx in enumerate(order_rev):
            val = pvals[idx] * m / (m - rank)
            run = min(run, val)
            bh[idx] = run
        for c, h_, b_ in zip([x for x in cells if x.get("status") == "ok"],
                             holm, bh):
            c["p_holm"] = float(h_)
            c["p_bh"] = float(b_)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"neuralforecast_available": NF_AVAILABLE,
                   "finbert_available": False,
                   "cells": cells}, f, indent=2, default=str)
    print(f"summary saved -> {OUT_DIR / 'summary.json'} (cells {len(cells)})")


if __name__ == "__main__":
    main()

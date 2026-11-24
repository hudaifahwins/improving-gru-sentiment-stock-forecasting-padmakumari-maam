"""Baseline rigor audit: Ridge & HistGradientBoostingRegressor.

Runs on the exact same non-overlapping h=3/5 windows as the GRU suite, with
identical train-only scalers, 70/15/15 and expanding WF, seeds 42/123/2026.
Reports price reconstruction, DirAcc vs majority base, k/3, Newey-West/HAC DM
(vs zero-return base and vs baseline GRU where available), turnover, net return
after 5bps/side, plus Holm and Benjamini-Hochberg adjusted p-values across all
tested cells. Cells with n < 30 are flagged.

Outputs: results/baseline_rigor_audit/summary.json (gitignored) + tables.
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

from run_consensus_conviction import (  # noqa: E402
    Cfg,
    evaluate,
    load_ohlcv,
    attach_local_sentiment,
    run_wf_folds,
)
from src.vol_gated_gru import prepare_split_data, hac_dm  # noqa: E402
from src.baseline import fit_scaler, compute_metrics  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "baseline_rigor_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2026)


def _windows_701515(cfg: Cfg, df: pd.DataFrame):
    split = prepare_split_data(df, features="price+tech+ohlcv_volnorm+sentiment",
                               horizon=cfg.horizon)
    return split


def fit_predict_flat(model, split):
    x_tr = split.x_train.reshape(split.x_train.shape[0], -1)
    x_te = split.x_test.reshape(split.x_test.shape[0], -1)
    model.fit(x_tr, split.y_train.ravel())
    return model.predict(x_te).reshape(-1)


def run_cell(ticker: str, bl: str, horizon: int, split_kind: str,
             seed: int) -> dict[str, Any]:
    df = load_ohlcv(ticker)
    df = attach_local_sentiment(df, ticker)
    cfg = Cfg(ticker, horizon, seed, "concat_sent_gru", split_kind, 0)
    if split_kind == "701515":
        split = _windows_701515(cfg, df)
        model = Ridge(alpha=1.0) if bl == "ridge" else HistGradientBoostingRegressor(
            random_state=seed, max_iter=300)
        p = fit_predict_flat(model, split)
        y = split.y_test[:, 0]
        prev = split.prev_test
    else:
        ys, ps, prevs = [], [], []
        for fold in run_wf_folds(cfg, df):
            model = Ridge(alpha=1.0) if bl == "ridge" else HistGradientBoostingRegressor(
                random_state=seed, max_iter=300)
            p = fit_predict_flat(model, fold)
            ys.append(fold.y_test[:, 0]); ps.append(p); prevs.append(fold.prev_test)
        y = np.concatenate(ys); p = np.concatenate(ps)
        prev = np.concatenate(prevs)
    e = (y - p) ** 2
    e_base = y ** 2
    full = evaluate(np.ones(len(y), bool), y, p, e_base)
    p_hat = prev * np.exp(p)
    pm = compute_metrics(prev * np.exp(y), p_hat, prev)
    return {
        "ticker": ticker, "baseline": bl, "horizon": horizon,
        "split": split_kind, "seed": seed, "full": full,
        "price_metrics": pm, "n": full["n"],
        "y": y.tolist(), "p": p.tolist(),
    }


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order):
        val = ps[idx] * (m - rank)
        running = min(running, val)
        adj[idx] = running
    return np.clip(adj, 0, 1).tolist()


def bh(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order):
        val = ps[idx] * m / (m - rank)
        running = min(running, val)
        adj[idx] = running
    return np.clip(adj, 0, 1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,005930.KS")
    ap.add_argument("--seeds", default="42,123,2026")
    args = ap.parse_args()
    tickers = args.tickers.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    cells: list[dict[str, Any]] = []
    ps = []
    for ticker in tickers:
        for bl in ("ridge", "gbr"):
            for horizon in (3, 5):
                for split_kind in ("701515", "wf"):
                    for seed in seeds:
                        t0 = time.perf_counter()
                        c = run_cell(ticker, bl, horizon, split_kind, seed)
                        cells.append(c)
                        ps.append(c["full"]["dm_p_vs_base"])
                        print(f"{ticker} {bl} h={horizon} {split_kind} "
                              f"seed={seed} n={c['n']} dir={c['full']['dir_acc']:.4f} "
                              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    holm_adj = holm(ps)
    bh_adj = bh(ps)
    for c, hp, bp in zip(cells, holm_adj, bh_adj):
        c["p_holm"] = float(hp)
        c["p_bh"] = float(bp)
        c["small_n_flag"] = c["n"] < 30
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(cells, f, indent=2, default=str)
    print(f"\nsummary saved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

"""Focused diagnostic experiment — hypothesis-driven.

Questions
---------
Q1: Factorize the VolGatedGRU + hysteresis failure:
    - loss: Huber-only vs Huber + directional margin
    - arch: GRU concat vs VolGatedGRU (gate)
    - hysteresis: off vs on (tau fit on val only)
    on AAPL and 005930.KS, h=3/5, 70/15/15 + expanding WF,
    seeds 42/123/2026.

Q2: Full-sample vs 3/3 seed-consensus subsets: n, coverage %, turnover,
    and 5bps/side net return — is the consensus edge selection-driven?

All features (price+tech+ohlcv_volnorm+sentiment) are causal; sentiment is
lagged 1 trading day; scalers fitted on train only; non-overlapping windows
(stride=horizon). Outputs go to results/research_diagnostics/ (gitignored).
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
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_consensus_conviction import (  # noqa: E402
    attach_local_sentiment,
    evaluate,
    load_ohlcv,
    run_wf_folds,
)
from src.vol_gated_gru import (  # noqa: E402
    COST_BPS_PER_SIDE,
    BaselineGRU,
    SplitData,
    VolGatedGRU,
    _causal_vol,
    fit_hysteresis_tau,
    hac_dm,
    predict,
    prepare_split_data,
)
from src.baseline import compute_metrics, fit_scaler, set_seed  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "research_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2026)


def train_diag(
    model: nn.Module,
    split: SplitData,
    *,
    gated: bool,
    loss: str,  # "huber" | "huber_dir"
    epochs: int,
    seed: int,
) -> None:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tr = torch.from_numpy(split.x_train)
    y_tr = torch.from_numpy(split.y_train)
    x_va = torch.from_numpy(split.x_val)
    y_va = torch.from_numpy(split.y_val)
    vol_tr = torch.from_numpy(_causal_vol(split.y_train[:, 0])).float()
    vol_va = torch.from_numpy(_causal_vol(split.y_val[:, 0])).float()
    k = 1.0 / float(np.std(split.y_train) + 1e-4)

    def step_loss(xb, yb, volb):
        if gated:
            s = xb[:, :, -split.n_sent:]
            main = xb[:, :, : -split.n_sent]
            out = model(main, s)[:, 0]
        else:
            out = model(xb)[:, 0]
        t = yb[:, 0]
        h = F.huber_loss(out / volb, t / volb, delta=1.0)
        if loss == "huber_dir":
            d = F.binary_cross_entropy_with_logits(k * out, (t > 0).float())
            return 0.5 * h + 0.5 * d
        return h

    best_val = float("inf")
    best_state = None
    bad = 0
    n = len(x_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss_v = step_loss(x_tr[idx], y_tr[idx], vol_tr[idx])
            loss_v.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = 0.0
            for i in range(0, len(x_va), 64):
                v += step_loss(x_va[i : i + 64], y_va[i : i + 64],
                               vol_va[i : i + 64]).item() * 64
        v /= len(x_va)
        if v < best_val - 1e-6:
            best_val, bad = v, 0
            best_state = {kk: vv.detach().clone()
                          for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    if best_state is not None:
        model.load_state_dict(best_state)


def run_cell(ticker: str, h: int, split_kind: str, arch: str, loss: str,
             hy: bool, seed: int, epochs: int) -> dict[str, Any]:
    df = load_ohlcv(ticker)
    df = attach_local_sentiment(df, ticker)
    gated = arch == "vgg"
    if split_kind == "701515":
        split = prepare_split_data(df, horizon=h)
        if not gated:
            # GRU concat uses all features
            model = BaselineGRU(input_size=split.x_train.shape[2])
        else:
            n_base = split.x_train.shape[2] - split.n_sent
            model = VolGatedGRU(input_size=n_base, n_sent=split.n_sent)
        train_diag(model, split, gated=gated, loss=loss, epochs=epochs, seed=seed)
        p = predict(model, split.x_test, use_sentiment=(not gated),
                    n_sent=split.n_sent, gated=gated)
        y = split.y_test[:, 0]
        prev = split.prev_test
        pred_val = predict(model, split.x_val, use_sentiment=(not gated),
                           n_sent=split.n_sent, gated=gated)
        y_val = split.y_val[:, 0]
    else:
        ys, ps, prevs, pred_vals, y_vals = [], [], [], [], []
        for fold in run_wf_folds(CfgStub(ticker, h, seed), df):
            if not gated:
                model = BaselineGRU(input_size=fold.x_train.shape[2])
            else:
                n_base = fold.x_train.shape[2] - fold.n_sent
                model = VolGatedGRU(input_size=n_base, n_sent=fold.n_sent)
            train_diag(model, fold, gated=gated, loss=loss, epochs=epochs, seed=seed)
            ps.append(predict(model, fold.x_test, use_sentiment=(not gated),
                              n_sent=fold.n_sent, gated=gated))
            ys.append(fold.y_test[:, 0])
            prevs.append(fold.prev_test)
            pred_vals.append(predict(model, fold.x_val, use_sentiment=(not gated),
                                     n_sent=fold.n_sent, gated=gated))
            y_vals.append(fold.y_val[:, 0])
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        prev = np.concatenate(prevs)
        pred_val = np.concatenate(pred_vals)
        y_val = np.concatenate(y_vals)

    e = (y - p) ** 2
    e_base = y ** 2
    full = evaluate(np.ones(len(y), bool), y, p, e_base)
    if hy:
        tau, _ = fit_hysteresis_tau(y_val, pred_val)
        mask = np.abs(p) > tau
    else:
        tau = 0.0
        mask = np.ones(len(y), bool)
    hy_eval = evaluate(mask, y, p, e_base)
    p_hat = prev * np.exp(p)
    pm = compute_metrics(prev * np.exp(y), p_hat, prev)
    return {
        "ticker": ticker, "horizon": h, "split": split_kind,
        "arch": arch, "loss": loss, "hysteresis": hy,
        "seed": seed, "n": int(len(y)),
        "full": full, "hysteresis_eval": hy_eval, "tau": tau,
        "price_metrics": pm,
        "y": y.tolist(), "p": p.tolist(),
    }


class CfgStub:
    def __init__(self, ticker, horizon, seed):
        self.ticker = ticker
        self.horizon = horizon
        self.seed = seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cells: list[dict[str, Any]] = []
    for ticker in ("AAPL", "005930.KS"):
        for h in (3, 5):
            for split_kind in ("701515", "wf"):
                for arch in ("gru", "vgg"):
                    for loss in ("huber", "huber_dir"):
                        for hy in (False, True):
                            for seed in seeds:
                                t0 = time.perf_counter()
                                c = run_cell(ticker, h, split_kind, arch, loss,
                                             hy, seed, args.epochs)
                                cells.append(c)
                                print(f"{ticker} h={h} {split_kind} {arch} "
                                      f"{loss} hy={int(hy)} seed={seed} "
                                      f"n={c['n']} ({time.perf_counter()-t0:.1f}s)",
                                      flush=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(cells, f, indent=2, default=str)
    print(f"summary saved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

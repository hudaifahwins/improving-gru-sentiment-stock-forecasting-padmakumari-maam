"""Run the constructive paper-upgrade suite (VolGatedGRU vs baselines).

Compares, for AAPL and 005930.KS at h=3 and h=5 (non-overlapping windows,
seeds 42 / 123 / 2026):

  1. Baseline GRU (price + tech)                    — MSE loss
  2. Standard concat-sentiment GRU                 — MSE loss
  3. VolGatedGRU + hysteresis false               — vol-normalized Huber
                                                    + directional margin, gate

Outputs
-------
- results/paper_upgrade/summary.json  (gitignored, all granular metrics)
- Full markdown tables printed to stdout (unrounded/4-decimal).

Usage
-----
    python scripts/run_paper_upgrade.py [--seeds 42,123,2026]
                                        [--horizons 3,5] [--epochs 150]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vol_gated_gru import (
    COST_BPS_PER_SIDE,
    SEEDS,
    BaselineGRU,
    SplitData,
    VolGatedGRU,
    fit_hysteresis_tau,
    hac_dm,
    load_ohlcv,
    prepare_split_data,
    predict,
    train_model,
)
from src.baseline import compute_metrics, set_seed

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "paper_upgrade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["baseline_gru", "concat_sentiment_gru", "vol_gated_gru"]


@dataclass
class RunConfig:
    ticker: str
    seed: int
    horizon: int
    epochs: int


def _split_for_model(split: SplitData, model: str) -> tuple[SplitData, bool]:
    """Feature columns: first k are base, last n_sent are sentiment.

    Standard concat sentiment GRU uses base+sentiment; baseline GRU base-only;
    vol-gated uses base branch + gate only on sentiment.
    """
    if model == "baseline_gru":
        return split, False
    return split, True


def run_single(cfg: RunConfig, df: pd.DataFrame) -> dict[str, Any]:
    set_seed(cfg.seed)
    split = prepare_split_data(
        df, features="price+tech+ohlcv_volnorm+sentiment", horizon=cfg.horizon
    )
    n_base = split.feature_names.index("sent_tone") if "sent_tone" in split.feature_names else len(split.feature_names)
    results: dict[str, Any] = {}

    for model_name in MODELS:
        use_sent = model_name != "baseline_gru"
        n_sent = split.n_sent

        if model_name == "baseline_gru":
            # base-only columns
            x_tr = split.x_train[:, :, : split.x_train.shape[2] - n_sent]
            x_va = split.x_val[:, :, : split.x_val.shape[2] - n_sent]
            x_te = split.x_test[:, :, : split.x_test.shape[2] - n_sent]
            split_b = SplitData(
                x_tr, split.y_train, split.p_train, split.prev_train,
                x_va, split.y_val, split.p_val, split.prev_val,
                x_te, split.y_test, split.p_test, split.prev_test,
                split.scaler, split.feature_names[:n_base], 0,
            )
            net = BaselineGRU(input_size=n_base)
            loss_kind = "mse"
        elif model_name == "concat_sentiment_gru":
            split_b = split
            net = BaselineGRU(input_size=split.x_train.shape[2])
            loss_kind = "mse"
        else:  # vol_gated_gru
            split_b = split
            net = VolGatedGRU(input_size=n_base, n_sent=n_sent)
            loss_kind = "vol_huber_margin"

        gated = model_name == "vol_gated_gru"
        hist = train_model(
            net, split_b, use_sentiment=use_sent, gated=gated,
            epochs=cfg.epochs, seed=cfg.seed, loss_kind=loss_kind,
        )
        pred = predict(net, split_b.x_test, use_sentiment=use_sent,
                       n_sent=split_b.n_sent, gated=gated)

        # price reconstruction
        p_hat = split_b.prev_test * np.exp(pred)
        price_metrics = compute_metrics(
            split_b.p_test, p_hat, split_b.prev_test
        )

        # directional metrics
        act_dir = np.sign(split_b.y_test[:, 0])
        pred_dir = np.sign(pred)
        dir_acc = float(np.mean(act_dir == pred_dir))
        majority = max(float(np.mean(act_dir > 0)), float(np.mean(act_dir < 0)))
        excess = dir_acc - majority

        # hysteresis tau on validation
        pred_val = predict(net, split_b.x_val, use_sentiment=use_sent,
                           n_sent=split_b.n_sent, gated=gated)
        tau, meta = fit_hysteresis_tau(split_b.y_val[:, 0], pred_val)

        # apply hysteresis to test
        sig = np.where(np.abs(pred) > tau, pred_dir, 0.0)
        cov = float(np.mean(np.abs(sig) > 0))
        r = np.expm1(split_b.y_test[:, 0])
        gross = float(np.sum(sig * r))
        flips = float(np.sum(np.abs(np.diff(sig)))) if len(sig) > 1 else 0.0
        turnover = flips / max(len(sig) - 1, 1)
        net_ret = gross - COST_BPS_PER_SIDE * flips

        # DM vs base rate and vs baseline GRU
        e_model = (split_b.y_test[:, 0] - pred) ** 2
        e_base = (split_b.y_test[:, 0] - np.zeros_like(pred)) ** 2
        e_ref = (split_b.y_test[:, 0] - np.sign(np.mean(split_b.y_test[:5]))) ** 2 if False else e_base
        z_base, p_base = hac_dm(e_model, e_base)
        z_ref, p_ref = (float("nan"), float("nan"))
        if "baseline_gru" in results:
            e_ref = results["baseline_gru"]["return_errors"]
            z_ref, p_ref = hac_dm(e_model, e_ref)

        results[model_name] = {
            "price_metrics": price_metrics,
            "dir_acc": dir_acc,
            "majority_base_rate": majority,
            "excess_dir_acc": excess,
            "hysteresis_tau": tau,
            "coverage": cov,
            "turnover": turnover,
            "gross_return": gross,
            "net_return_5bps": net_ret,
            "dm_vs_base": z_base,
            "dm_p_vs_base": p_base,
            "dm_vs_baseline_gru": z_ref,
            "dm_p_vs_baseline_gru": p_ref,
            "return_errors": e_model.tolist(),
            "best_epoch": hist["best_epoch"],
            "epochs_run": hist["epochs_run"],
            "n_test": int(len(sig)),
            "seed": cfg.seed,
        }
    collapse = sum(1 for m in MODELS if results[m]["dir_acc"] <= results[m]["majority_base_rate"])
    return {"config": asdict(cfg), "results": results, "seed_collapse_k_of_3": collapse}


def fmt(v: float, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}"


def _meanstd(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    if np.all(np.isnan(a)):
        return "—"
    return f"{np.nanmean(a):.4f} ± {np.nanstd(a):.4f}"


def build_table(rows: list[dict[str, Any]], horizon: int, ticker: str) -> str:
    lines = [f"### {ticker} — h={horizon} (non-overlapping; mean ± std over seeds 42/123/2026)", ""]
    header = ("| model | MAE | RMSE | nRMSE | MAPE | DirAcc | base rate | excess | "
              "seed excess≤0 | DM(base) p | DM(baseline) p | coverage | turnover | net ret 5bps |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines += [header, sep]
    for model_name in MODELS:
        ms = [r["results"][model_name] for r in rows]
        pm = [m["price_metrics"] for m in ms]
        dirs = [m["dir_acc"] for m in ms]
        bases = [m["majority_base_rate"] for m in ms]
        excess = [m["excess_dir_acc"] for m in ms]
        n_collapse = sum(1 for e in excess if e <= 0.0)
        p_base = [m["dm_p_vs_base"] for m in ms]
        p_ref = [m["dm_p_vs_baseline_gru"] for m in ms]
        cov = [m["coverage"] for m in ms]
        to = [m["turnover"] for m in ms]
        net = [m["net_return_5bps"] for m in ms]
        lines.append(
            f"| {model_name} | {_meanstd([x['mae'] for x in pm])} | "
            f"{_meanstd([x['rmse'] for x in pm])} | {_meanstd([x['nrmse'] for x in pm])} | "
            f"{_meanstd([x['mape'] for x in pm])} | {_meanstd(dirs)} | {_meanstd(bases)} | "
            f"{_meanstd(excess)} | {n_collapse}/3 | {_meanstd(p_base)} | {_meanstd(p_ref)} | "
            f"{_meanstd(cov)} | {_meanstd(to)} | {_meanstd(net)} |"
        )
    lines += ["", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--horizons", default="3,5")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tickers", default="AAPL,005930.KS")
    ap.add_argument("--report-only", action="store_true",
                    help="render aggregated tables from results/paper_upgrade/summary.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]
    tickers = args.tickers.split(",")

    if args.report_only:
        if not (OUT_DIR / "summary.json").exists():
            raise SystemExit("summary.json not found — run without --report-only first")
        loaded = json.load(open(OUT_DIR / "summary.json"))
        all_runs = loaded["runs"]
        markdown = ["# Paper Upgrade — VolGatedGRU vs baselines (verified, aggregated)", ""]
        for ticker in tickers:
            for horizon in horizons:
                runs = [r for r in all_runs
                        if r["config"]["ticker"] == ticker and r["config"]["horizon"] == horizon]
                if runs:
                    markdown.append(build_table(runs, horizon, ticker))
        print("\n".join(markdown))
        return

    all_runs: list[dict[str, Any]] = []
    markdown = ["# Paper Upgrade — VolGatedGRU vs baselines (verified)", ""]
    for ticker in tickers:
        df = load_ohlcv(ticker)
        for horizon in horizons:
            runs = []
            for seed in seeds:
                print(f"running {ticker} h={horizon} seed={seed} …", flush=True)
                t0 = time.perf_counter()
                runs.append(run_single(RunConfig(ticker, seed, horizon, args.epochs), df))
                print(f"  done in {time.perf_counter()-t0:.1f}s")
            all_runs.extend(runs)
            markdown.append(build_table(runs, horizon, ticker))

    summary = {"runs": all_runs}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n".join(markdown))
    print(f"\nsummary saved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()

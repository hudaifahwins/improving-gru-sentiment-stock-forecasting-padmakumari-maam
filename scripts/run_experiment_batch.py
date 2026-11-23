"""Sequential driver for the E1-E8 wire-up experiment batch (leak-free).

Every configuration uses the identical chronological 70/15/15 split per seed,
train-only MinMax scaling, backward-looking 30-day windows, and the 1-day-lagged
GDELT tone/count features from data/prices_sentiment_<ticker>.csv. Arms:

  A0  verified-baseline reproduction (10y yfinance window, price-only)
  A   h=1 overlap MSE feature ablation (price / +tech / +ohlcv_volnorm /
      +sentiment / +ohlcv_volnorm+sentiment)
  B   h=1 loss & evaluation variants (hybrid, dir_margin, causal vol-deadzone)
  C0  h=3/5 non-overlap price-only reference
  C   h=3/5 non-overlap feature ablation (tech / volnorm / full stack)
  D   h=3/5 overlap full-stack reference
  E   h=1 GBR comparators (price / full stack)
  F   return-target harness (10y window, price-only; log/logvol x mse/hybrid/
      dir_margin) — mirrors the multifront protocol
  G   price-level baseline.py reproduction (nRMSE anchor vs 0.038 / paper 0.048)

Writes one JSON per run (via the harnesses) plus per-run logs into
results/batch_e1e8/ and a resumable manifest (batch_manifest.jsonl).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "batch_e1e8"
LOGS = OUT / "logs"
MANIFEST = OUT / "batch_manifest.jsonl"
SEEDS = [42, 43, 44]
TICKERS = ["AAPL", "005930.KS"]
CSV = {t: f"data/prices_sentiment_{t}.csv" for t in TICKERS}
SCOLS = "Sentiment,NewsCount"
FULL = "price+tech+ohlcv_volnorm+sentiment"


def horizon(ticker, *, csv_mode=True, h=1, features="price", loss="mse",
            stride="overlap", baseline="ridge", vol_deadzone=0.0, seed=42,
            epochs=100):
    argv = [sys.executable, "-m", "src.horizon_experiment"]
    if csv_mode:
        argv += ["--csv", CSV[ticker], "--sentiment-cols", SCOLS]
    else:
        argv += ["--ticker", ticker]
    argv += ["--horizon", str(h), "--features", features, "--loss", loss,
             "--stride", stride, "--baseline", baseline, "--seed", str(seed),
             "--epochs", str(epochs), "--out-dir", str(OUT)]
    if vol_deadzone > 0:
        argv += ["--vol-deadzone", str(vol_deadzone)]
    return argv


def return_exp(ticker, *, rt="log", loss="mse", seed=42, epochs=100):
    return [sys.executable, "-m", "src.return_experiment", "--ticker", ticker,
            "--return-type", rt, "--loss", loss, "--baseline", "ridge",
            "--seed", str(seed), "--epochs", str(epochs), "--out-dir", str(OUT)]


def baseline_run(ticker, seed):
    return [sys.executable, "-m", "src.baseline", "--ticker", ticker,
            "--model", "gru", "--hidden-size", "50", "--layers", "1",
            "--epochs", "100", "--seed", str(seed), "--out-dir", str(OUT)]


def build_jobs():
    jobs = []

    def add(arm, label, argv):
        jobs.append({"arm": arm, "label": label, "argv": argv})

    for T in TICKERS:
        for s in SEEDS:
            add("A0", f"A0__{T}__h1_price_mse_overlap_ridge__s{s}",
                horizon(T, csv_mode=False, h=1, features="price", loss="mse",
                        stride="overlap", baseline="ridge", seed=s))
            for feat in ["price", "price+tech", "price+tech+ohlcv_volnorm",
                         "price+tech+sentiment", FULL]:
                add("A", f"A__{T}__h1_{feat}__mse_overlap_ridge__s{s}",
                    horizon(T, h=1, features=feat, loss="mse", stride="overlap",
                            baseline="ridge", seed=s))
            add("B", f"B__{T}__h1_tech_hybrid__s{s}",
                horizon(T, h=1, features="price+tech", loss="hybrid",
                        stride="overlap", baseline="ridge", seed=s))
            add("B", f"B__{T}__h1_full_hybrid__s{s}",
                horizon(T, h=1, features=FULL, loss="hybrid", stride="overlap",
                        baseline="ridge", seed=s))
            add("B", f"B__{T}__h1_full_dir_margin__s{s}",
                horizon(T, h=1, features=FULL, loss="dir_margin", stride="overlap",
                        baseline="ridge", seed=s))
            add("B", f"B__{T}__h1_full_voldead1.0__s{s}",
                horizon(T, h=1, features=FULL, loss="mse", stride="overlap",
                        baseline="ridge", vol_deadzone=1.0, seed=s))
            add("B", f"B__{T}__h1_tech_voldead1.0__s{s}",
                horizon(T, h=1, features="price+tech", loss="mse", stride="overlap",
                        baseline="ridge", vol_deadzone=1.0, seed=s))
            for h in (3, 5):
                add("C0", f"C0__{T}__h{h}_price_mse_nonoverlap_ridge__s{s}",
                    horizon(T, h=h, features="price", loss="mse",
                            stride="nonoverlap", baseline="ridge", seed=s))
                for feat in ["price+tech", "price+tech+ohlcv_volnorm", FULL]:
                    add("C", f"C__{T}__h{h}_{feat}__mse_nonoverlap_ridge__s{s}",
                        horizon(T, h=h, features=feat, loss="mse",
                                stride="nonoverlap", baseline="ridge", seed=s))
            for h in (3, 5):
                add("D", f"D__{T}__h{h}_full_mse_overlap_ridge__s{s}",
                    horizon(T, h=h, features=FULL, loss="mse", stride="overlap",
                            baseline="ridge", seed=s))
            for feat in ["price", FULL]:
                add("E", f"E__{T}__h1_{feat}__mse_overlap_gbr__s{s}",
                    horizon(T, h=1, features=feat, loss="mse", stride="overlap",
                            baseline="gbr", seed=s))
            for rt, loss in [("log", "mse"), ("log", "hybrid"),
                             ("logvol", "mse"), ("logvol", "dir_margin")]:
                add("F", f"F__{T}__{rt}_{loss}_ridge__s{s}",
                    return_exp(T, rt=rt, loss=loss, seed=s))
            add("G", f"G__{T}__pricelevel_gru50x1__s{s}", baseline_run(T, s))
    return jobs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            try:
                done.add(json.loads(line)["label"])
            except Exception:
                pass
    jobs = build_jobs()
    env = dict(os.environ, OMP_NUM_THREADS="2")
    t0 = time.time()
    n_ok = n_fail = n_skip = 0
    with MANIFEST.open("a") as mf:
        for i, job in enumerate(jobs, 1):
            if job["label"] in done:
                n_skip += 1
                continue
            log = LOGS / f"{job['label']}.log"
            with log.open("w") as lf:
                proc = subprocess.run(job["argv"], cwd=ROOT, env=env,
                                      stdout=lf, stderr=subprocess.STDOUT)
            rec = {"label": job["label"], "arm": job["arm"],
                   "argv": job["argv"], "rc": proc.returncode, "log": str(log)}
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            if proc.returncode == 0:
                n_ok += 1
            else:
                n_fail += 1
                print(f"[FAIL rc={proc.returncode}] {job['label']}", flush=True)
            print(f"[{i}/{len(jobs)}] {job['label']} rc={proc.returncode} "
                  f"elapsed={time.time() - t0:.0f}s ok={n_ok} fail={n_fail} "
                  f"skip={n_skip}", flush=True)
    print(f"BATCH DONE ok={n_ok} fail={n_fail} skip={n_skip} "
          f"elapsed={(time.time() - t0) / 60:.1f}min")


if __name__ == "__main__":
    main()

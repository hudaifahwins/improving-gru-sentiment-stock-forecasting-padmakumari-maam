"""Aggregate results/batch_e1e8 into SUMMARY.md + batch_summary.json.

Reads every horizon/return/baseline JSON in the batch dir, groups by
(ticker, horizon, features, loss, stride, vol_deadzone, baseline), reports
mean ± std across seeds, and runs *window-level* tests on the per-window dump:
* McNemar-style sign test (exact binomial) GRU vs causal momentum,
* Wilcoxon signed-rank GRU vs Ridge on paired |return errors|,
* Fisher combination of per-seed p-values.

Seed-level means are effect sizes; window-level tests carry the statistical
evidence (n = disjoint test windows). Nothing is dropped or cherry-picked:
every JSON found in the batch dir is aggregated.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BATCH = Path("results/batch_e1e8")
TICKERS = ["AAPL", "005930.KS"]


def ticker_of(d: dict) -> str:
    m = re.search(r"(AAPL|005930\.KS)", str(d.get("data_source", "")))
    return m.group(1) if m else "unknown"


def ms(vals):
    v = [x for x in vals if x is not None and (not isinstance(x, float) or not np.isnan(x))]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def fmt(m, nd=4):
    if m is None:
        return "—"
    return f"{m['mean']:.{nd}f} ± {m['std']:.{nd}f}" if m["n"] > 1 else f"{m['mean']:.{nd}f}"


def window_tests(w):
    """GRU vs momentum sign test + GRU vs Ridge Wilcoxon on one run.

    Also computes the majority-sign (no-skill) base rate from the actual test
    targets, a constant-sign collapse flag, and excess DirAcc over the base
    rate — the minimal honest comparators for directional accuracy.
    """
    y = np.asarray(w["y"], float)
    p = np.asarray(w["pred"], float)
    psign = np.sign(p)
    gru_hit = psign == np.sign(y)
    out = {"n": int(len(y))}
    out["collapsed"] = bool(len(np.unique(psign)) == 1)
    out["base_up"] = float((y > 0).mean())
    out["base_down"] = float((y < 0).mean())
    base_maj = max(out["base_up"], out["base_down"])
    out["base_majority"] = base_maj
    out["excess_over_base"] = float(gru_hit.mean() - base_maj)
    msign = np.array([np.nan if v is None else v for v in w["momentum_sign"]], float)
    valid = ~np.isnan(msign)
    if valid.sum() > 0:
        mom_hit = msign[valid] == np.sign(y)[valid]
        g = gru_hit[valid]
        b = int((g & ~mom_hit).sum())   # GRU right, momentum wrong
        c = int((~g & mom_hit).sum())   # GRU wrong, momentum right
        out["mom_n"] = int(valid.sum())
        out["mom_b"] = b
        out["mom_c"] = c
        out["mom_excess"] = float(mom_hit.mean() - base_maj)
        if b + c > 0:
            out["mom_p"] = float(stats.binomtest(b, b + c, 0.5).pvalue)
    rp = w.get("ridge_pred")
    if rp is not None:
        rp = np.asarray(rp, float)
        if len(rp) == len(y) and len(y) > 5:
            eg = np.abs(y - p)
            er = np.abs(y - rp)
            out["ridge_mae_gru"] = float(eg.mean())
            out["ridge_mae_ridge"] = float(er.mean())
            try:
                out["ridge_wilcoxon_p"] = float(stats.wilcoxon(eg, er).pvalue)
            except Exception:
                out["ridge_wilcoxon_p"] = None
    return out


def combine_ps(ps):
    ps = [p for p in ps if p is not None]
    if not ps:
        return None
    if len(ps) == 1:
        return ps[0]
    return float(stats.combine_pvalues(ps, method="fisher").pvalue)


def main() -> None:
    horizon_groups = defaultdict(list)   # key -> list of payloads
    return_groups = defaultdict(list)
    baseline_runs = []
    for f in sorted(BATCH.glob("*.json")):
        if f.name == "batch_summary.json":
            continue
        d = json.loads(f.read_text())
        cfg = d.get("config", {})
        if "h" in cfg:  # horizon_experiment
            h = cfg["h"]
            src = str(d.get("data_source", ""))
            window = "yf10y" if src.startswith("yfinance") else "merged"
            key = (ticker_of(d), window, h["horizon"], h["features"], h["loss"],
                   h["stride"], h.get("vol_deadzone", 0.0), h["baseline"])
            rec = {"seed": h["seed"], "payload": d}
            horizon_groups[key].append(rec)
        elif "experiment" in cfg:  # return_experiment
            e = cfg["experiment"]
            key = (ticker_of(d), e["return_type"], e["loss"])
            return_groups[key].append({"seed": cfg["model"]["seed"], "payload": d})
        elif "model" in cfg and "run" in cfg:  # baseline.py
            baseline_runs.append(d)

    # ---------------------------------------------------------------- horizon
    summary = {}
    for key, recs in sorted(horizon_groups.items()):
        T, window, h, feat, loss, stride, vd, base = key
        g = summary.setdefault(key, {"ticker": T, "window": window, "h": h,
                                     "features": feat, "loss": loss, "stride": stride,
                                     "vol_deadzone": vd, "baseline": base,
                                     "seeds": []})
        for r in sorted(recs, key=lambda r: r["seed"]):
            d = r["payload"]
            gru = d["results"]["gru"]
            rm = gru["return_metrics"]
            pm = gru["price_metrics"]
            mom = d["results"].get("momentum_baseline", {})
            ridge = d["results"].get("ridge", {})
            gbr = d["results"].get("gbr", {})
            wt = window_tests(d["results"]["windows"])
            g["seeds"].append({
                "seed": r["seed"],
                "dir_acc": rm["directional_accuracy"],
                "dir_acc_strict": rm["directional_accuracy_strict"],
                "deadzone_share": rm["deadzone_share"],
                "price_mae": pm["mae"], "price_nrmse": pm["nrmse"],
                "return_mae": rm["return_mae"],
                "momentum_dir_acc": mom.get("directional_accuracy"),
                "momentum_n_valid": mom.get("n_valid"),
                "ridge_dir_acc": ridge.get("return_metrics", {}).get("directional_accuracy"),
                "gbr_dir_acc": gbr.get("return_metrics", {}).get("directional_accuracy"),
                "best_epoch": gru["best_epoch"], "epochs_run": gru["epochs_run"],
                "train_time_s": gru["train_time_s"],
                "vol_filtered_share": rm.get("vol_filtered_share"),
                "dir_acc_vol_filtered": rm.get("dir_acc_vol_filtered"),
                "n_test_windows": wt["n"],
                "base_up": wt["base_up"], "base_majority": wt["base_majority"],
                "excess_over_base": wt["excess_over_base"],
                "collapsed": wt["collapsed"],
                "mom_excess": wt.get("mom_excess"),
                "mom_n": wt.get("mom_n"), "mom_b": wt.get("mom_b"),
                "mom_c": wt.get("mom_c"), "mom_p": wt.get("mom_p"),
                "ridge_mae_gru": wt.get("ridge_mae_gru"),
                "ridge_mae_ridge": wt.get("ridge_mae_ridge"),
                "ridge_wilcoxon_p": wt.get("ridge_wilcoxon_p"),
                "feature_list": d.get("features"),
                "split": d.get("split"),
            })
        S = g["seeds"]
        g["dir_acc"] = ms([s["dir_acc"] for s in S])
        g["dir_acc_strict"] = ms([s["dir_acc_strict"] for s in S])
        g["price_mae"] = ms([s["price_mae"] for s in S])
        g["price_nrmse"] = ms([s["price_nrmse"] for s in S])
        g["return_mae"] = ms([s["return_mae"] for s in S])
        g["momentum_dir_acc"] = ms([s["momentum_dir_acc"] for s in S])
        g["ridge_dir_acc"] = ms([s["ridge_dir_acc"] for s in S])
        g["gbr_dir_acc"] = ms([s["gbr_dir_acc"] for s in S])
        g["mom_p_fisher"] = combine_ps([s["mom_p"] for s in S])
        g["mom_b_total"] = sum(s["mom_b"] or 0 for s in S)
        g["mom_c_total"] = sum(s["mom_c"] or 0 for s in S)
        g["ridge_wilcoxon_p_fisher"] = combine_ps([s["ridge_wilcoxon_p"] for s in S])
        g["n_test_windows"] = ms([s["n_test_windows"] for s in S])
        g["mom_n"] = ms([s["mom_n"] for s in S])
        g["vol_filtered_share"] = ms([s["vol_filtered_share"] for s in S])
        g["dir_acc_vol_filtered"] = ms([s["dir_acc_vol_filtered"] for s in S])
        g["base_up"] = ms([s["base_up"] for s in S])
        g["base_majority"] = ms([s["base_majority"] for s in S])
        g["excess_over_base"] = ms([s["excess_over_base"] for s in S])
        g["mom_excess"] = ms([s["mom_excess"] for s in S])
        g["collapsed_any"] = any(s["collapsed"] for s in S)
        g["collapsed_all"] = all(s["collapsed"] for s in S)
        g["collapsed_seeds"] = [s["seed"] for s in S if s["collapsed"]]
        g["best_epoch"] = ms([s["best_epoch"] for s in S])
        g["train_time_s"] = ms([s["train_time_s"] for s in S])
        g["feature_list"] = S[0]["feature_list"]
        g["split"] = S[0]["split"]

    # ------------------------------------------------------------------ return
    return_summary = {}
    for key, recs in sorted(return_groups.items()):
        T, rt, loss = key
        g = return_summary.setdefault(key, {"ticker": T, "return_type": rt,
                                            "loss": loss, "seeds": []})
        for r in sorted(recs, key=lambda r: r["seed"]):
            d = r["payload"]
            res = d["results"]
            raw = res["raw_price_gru"]["price_metrics"]
            ret = res["return_gru"]
            rpm = ret["price_metrics_reconstructed"]
            rrm = ret["return_metrics"]
            per = res["persistence"]["price_metrics"]
            g["seeds"].append({
                "seed": r["seed"],
                "raw_dir": raw["directional_accuracy"], "raw_nrmse": raw["nrmse"],
                "ret_dir": rrm["directional_accuracy"],
                "ret_strict": rrm["directional_accuracy_strict"],
                "ret_deadzone": rrm["deadzone_share"],
                "ret_mae": rrm["return_mae"], "ret_rmse": rrm["return_rmse"],
                "ret_price_mae": rpm["mae"], "ret_price_nrmse": rpm["nrmse"],
                "persist_dir": per["directional_accuracy"],
                "cum_buy_hold": rrm["cumulative_return_buy_and_hold"],
                "cum_sign": rrm["cumulative_return_sign_strategy"],
                "ridge_dir": (res.get("ridge_baseline", {})
                              .get("return_metrics", {})
                              .get("directional_accuracy")),
                "best_epoch": ret["best_epoch"], "train_time_s": ret["train_time_s"],
            })
        S = g["seeds"]
        for k in ["raw_dir", "raw_nrmse", "ret_dir", "ret_strict", "ret_deadzone",
                  "ret_mae", "ret_rmse", "ret_price_mae", "ret_price_nrmse",
                  "persist_dir", "cum_buy_hold", "cum_sign", "ridge_dir",
                  "best_epoch", "train_time_s"]:
            g[k] = ms([s[k] for s in S])

    # -------------------------------------------------------------- baseline.py
    base_summary = {}
    for d in baseline_runs:
        T = ticker_of(d)
        g = base_summary.setdefault(T, {"seeds": []})
        m = d["results"]["model"]["metrics"]
        g["seeds"].append({"seed": d["config"]["model"]["seed"], "nrmse": m["nrmse"],
                           "mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"],
                           "r2": m["r2"], "dir_acc": m["directional_accuracy"],
                           "n_rows": d["n_rows"]})
    for T, g in base_summary.items():
        for k in ["nrmse", "mae", "rmse", "mape", "r2", "dir_acc"]:
            g[k] = ms([s[k] for s in g["seeds"]])
        g["n_rows"] = g["seeds"][0]["n_rows"]

    out = {"horizon": {str(k): v for k, v in summary.items()},
           "return": {str(k): v for k, v in return_summary.items()},
           "baseline": base_summary}
    (BATCH / "batch_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"aggregated {len(summary)} horizon groups, {len(return_summary)} return groups, "
          f"{len(base_summary)} baseline tickers -> {BATCH / 'batch_summary.json'}")


if __name__ == "__main__":
    main()

"""Render findings.md from results/opensource_benchmark/summary.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "opensource_benchmark"
SUMMARY = OUT / "summary.json"


def ms(vals, d=4):
    a = np.asarray(vals, dtype=float)
    if np.all(np.isnan(a)):
        return "—"
    return f"{np.nanmean(a):.{d}f} ± {np.nanstd(a):.{d}f}"


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit("summary.json missing — run run_opensource_benchmark.py first")
    data = json.load(open(SUMMARY))
    cells = [c for c in data["cells"] if c.get("status") == "ok"]
    lines = [
        "# Open-Source Benchmark — findings (verified, seeds 42/123/2026)", "",
        f"- neuralforecast available: {data.get('neuralforecast_available')} "
        f"(NBEATS / NHITS, v3.3.1)",
        f"- FinBERT available: {data.get('finbert_available')} — not cached; "
        f"Loughran–McDonald used for firm-level text instead (no fabricated scores)",
        f"- Protocol: expanding rolling origins, non-overlapping h steps, train-only "
        f"scalers, strictly causal features, 5bps/side cost, Newey-West/HAC DM + Holm/BH.",
        "",
        "| ticker | h | family | n | MAE | RMSE | nRMSE | MAPE | DirAcc | excess | DMp(base) | Holm | BH | turnover | net5bps |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    # aggregate by key
    from collections import defaultdict

    groups = defaultdict(list)
    for c in cells:
        groups[(c["ticker"], c["horizon"], c["family"])].append(c)

    for (t, h, fam) in sorted(groups):
        g = groups[(t, h, fam)]
        pm = [c["price_metrics"] for c in g]
        lines.append(
            f"| {t} | {h} | {fam} | {g[0]['n']} | "
            f"{ms([x['mae'] for x in pm])} | {ms([x['rmse'] for x in pm])} | "
            f"{ms([x['nrmse'] for x in pm])} | {ms([x['mape'] for x in pm])} | "
            f"{ms([c['dir_acc'] for c in g])} | "
            f"{ms([c['excess_dir_acc'] for c in g])} | "
            f"{ms([c['dm_p_vs_base'] for c in g])} | "
            f"{ms([c.get('p_holm') for c in g])} | "
            f"{ms([c.get('p_bh') for c in g])} | "
            f"{ms([c['turnover'] for c in g])} | "
            f"{ms([c['net_return_5bps'] for c in g])} |"
        )
    lines += [
        "",
        "## Honest notes",
        "- NBEATS / NHITS are UNIVARIATE close-price models (they cannot consume our",
        "  windowed exogenous tensor without look-ahead risk); the GRU / VolGatedGRU /",
        "  Ridge / HistGBR rows include causal covariates. That is the main apples-to-",
        "  oranges caveat when comparing families.",
        "- NeuralForecast models are fit per origin; runtime is much higher than the",
        "  sklearn / GRU families.",
        "- Holm/BH adjustment is applied across all tested cells (not per family).",
        "- Cells with n < 30 must be read as diagnostic only.",
    ]
    (OUT / "findings.md").write_text("\n".join(lines) + "\n")
    print(f"findings.md written -> {OUT / 'findings.md'}")


if __name__ == "__main__":
    main()

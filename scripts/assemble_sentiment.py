#!/usr/bin/env python3
"""Merge per-ticker sentiment-ablation parts and recompute Holm per family.

Usage: python3 scripts/assemble_sentiment.py <parts_glob> <outdir>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_decisive_multiday as base  # noqa: E402


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("outdir")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no summaries match {args.pattern}")
    cells, paired, ics, protocol = [], [], [], None
    for fp in files:
        p = json.load(open(fp))
        protocol = protocol or p["protocol"]
        cells.extend(p["cells"]); paired.extend(p["paired"]); ics.extend(p["sentiment_ic"])

    for c, a in zip(cells, base.holm([c["dm_p_vs_hist_mean"] for c in cells])):
        c["p_holm_vs_hist_mean"] = float(a)
    for p, a in zip(paired, base.holm([p["dm_p_sent_vs_price"] for p in paired])):
        p["p_holm_sent_vs_price"] = float(a)

    order = {"hist_mean": 0, "ridge_price": 1, "ridge_sent": 2,
             "gru_price": 3, "gru_sent": 4}
    cells.sort(key=lambda c: (c["ticker"], c["horizon"], order.get(c["model"], 9)))
    paired.sort(key=lambda c: (c["ticker"], c["horizon"], c["comparison"]))

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"protocol": protocol, "sentiment_ic": ics, "cells": cells,
               "paired": paired}, open(out / "summary.json", "w"), indent=2)

    L = ["# Sentiment ablation — does news sentiment add value?", "",
         f"- paired design, {len(files)} tickers; disjoint origins; 5 bps/side; "
         f"seeds {protocol['seeds']}; up to {protocol['origins_cap']} origins/cell; "
         "sentiment features = level + 3-day mean (already lagged 1 day)", "",
         "## Unconditional sentiment content", ""]
    L += md_table(["ticker", "h", "n", "Pearson r", "p", "Spearman r", "p"],
                  [[i["ticker"], str(i["horizon"]), str(i["n"]),
                    f"{i['pearson_r']:+.4f}", f"{i['pearson_p']:.4g}",
                    f"{i['spearman_r']:+.4f}", f"{i['spearman_p']:.4g}"] for i in ics])
    L += ["", "## Arm-level results (family A: vs historical mean)", ""]
    L += md_table(["ticker", "h", "model", "n", "DirAcc", "majority", "excess",
                   "DM p", "Holm", "net5bps", "seed sd"],
                  [[c["ticker"], str(c["horizon"]), c["model"], str(c["n"]),
                    f"{c['dir_acc']:.4f}", f"{c['majority_base']:.4f}",
                    f"{c['excess']:+.4f}", f"{c['dm_p_vs_hist_mean']:.4g}",
                    f"{c['p_holm_vs_hist_mean']:.4g}", f"{c['net_return_5bps']:+.4f}",
                    f"{c['seed_spread']:.3f}" if c["seed_spread"] is not None else "-"]
                   for c in cells])
    L += ["", "## Paired test (family B: sentiment vs its price-only twin)", ""]
    L += md_table(["ticker", "h", "comparison", "n", "DirAcc sent", "DirAcc price",
                   "delta", "MSE sent", "MSE price", "DM p", "Holm", "McNemar p"],
                  [[p["ticker"], str(p["horizon"]), p["comparison"], str(p["n"]),
                    f"{p['dir_acc_sent']:.4f}", f"{p['dir_acc_price']:.4f}",
                    f"{p['delta_dir_acc']:+.4f}", f"{p['mse_sent']:.6f}",
                    f"{p['mse_price']:.6f}", f"{p['dm_p_sent_vs_price']:.4g}",
                    f"{p['p_holm_sent_vs_price']:.4g}", f"{p['mcnemar_p']:.4g}"]
                   for p in paired])
    (out / "findings.md").write_text("\n".join(L) + "\n")
    print(f"merged {len(files)} parts: {len(cells)} cells, {len(paired)} paired -> {out}")


if __name__ == "__main__":
    main()

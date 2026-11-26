#!/usr/bin/env python3
"""Assemble per-ticker decisive runs into one Holm-corrected result set.

The decisive harness is run once per ticker (so the parts can execute in
parallel). Holm correction must be applied across *all* cells, so it is
recomputed here on the merged set.

Usage: python3 scripts/assemble_decisive.py <parts_glob> <outdir>
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("outdir")
    ap.add_argument("--title", default="Decisive multi-day evaluation — "
                                      "non-overlapping, cost-aware")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no summaries match {args.pattern}")
    cells, protocol = [], None
    for fp in files:
        payload = json.load(open(fp))
        protocol = protocol or payload.get("protocol", {})
        cells.extend(payload["cells"])
    cells.sort(key=lambda c: (c["ticker"], c["horizon"], c["model"]))

    adj = base.holm([c["dm_p_vs_hist_mean"] for c in cells])
    for c, a in zip(cells, adj):
        c["p_holm"] = float(a)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"protocol": protocol, "cells": cells},
              open(out / "summary.json", "w"), indent=2)

    lines = [f"# {args.title}", "",
             "- disjoint evaluation windows; 5 bps/side; "
             f"seeds {protocol.get('seeds')}; up to "
             f"{protocol.get('origins_cap')} origins per cell; "
             f"lookback {protocol.get('lookback')}",
             "- DM = Diebold–Mariano (HAC) vs historical-mean forecast; "
             "Holm across all cells", "",
             "| ticker | h | model | n | DirAcc | majority | excess | DM p | Holm | net5bps | turnover |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(
            f"| {c['ticker']} | {c['horizon']} | {c['model']} | {c['n']} | "
            f"{c['dir_acc']:.4f} | {c['majority_base']:.4f} | "
            f"{c['excess']:+.4f} | {c['dm_p_vs_hist_mean']:.4g} | "
            f"{c['p_holm']:.4g} | {c['net_return_5bps']:+.4f} | "
            f"{c['turnover']:.2f} |")
    (out / "findings.md").write_text("\n".join(lines) + "\n")
    print(f"assembled {len(files)} parts, {len(cells)} cells -> {out}")


if __name__ == "__main__":
    main()

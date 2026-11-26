#!/usr/bin/env python3
"""Figures for the results pack, generated from the saved summaries.

Outputs (docs/assets/):
  fig_excess_diracc.png    mean excess directional accuracy by model x horizon,
                           with each market's cell overlaid as a dot
  fig_net_vs_turnover.png  net return after 5 bps/side vs turnover, by model
  fig_sentiment_delta.png  paired sentiment-minus-price directional accuracy

Only plots recorded numbers; no smoothing, no re-fitting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
DEC = ROOT / "results" / "decisive_multiday" / "summary.json"
SENT = ROOT / "results" / "sentiment_ablation" / "summary.json"

MODEL_ORDER = ["hist_mean", "majority", "persistence", "momentum", "ridge", "gru"]
COLORS = dict(zip(MODEL_ORDER, ["#9e9e9e", "#bdbdbd", "#c8b6e2", "#f0a868",
                                "#4c9f70", "#2f6db3"]))


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the experiments first")
    return json.load(open(path))


def fig_excess(dec: dict) -> None:
    # persistence is a degenerate baseline (predicts zero -> never right on sign)
    cells = [c for c in dec["cells"] if c["model"] not in ("majority", "persistence")]
    horizons = sorted({c["horizon"] for c in cells})
    models = [m for m in MODEL_ORDER if any(c["model"] == m for c in cells)]
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4.2),
                             sharey=True)
    if len(horizons) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for ax, h in zip(axes, horizons):
        data, labels = [], []
        for m in models:
            vals = [c["excess"] for c in cells if c["horizon"] == h and c["model"] == m]
            if vals:
                data.append(vals); labels.append(m)
        means = [np.mean(v) for v in data]
        ax.bar(range(len(data)), means, color=[COLORS[l] for l in labels],
               alpha=0.85, width=0.65)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=16,
                       color="#1a1a1a", alpha=0.55, zorder=3)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=40,
                           ha="right", fontsize=8)
        ax.set_title(f"h = {h} day(s)", fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].set_ylabel("excess directional accuracy\n(DirAcc − majority base)")
    fig.suptitle("No arm reliably beats the majority base rate — "
                 f"5 markets, {len(cells)} evaluated arms", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS / "fig_excess_diracc.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_net(dec: dict) -> None:
    cells = [c for c in dec["cells"] if c["model"] in ("ridge", "gru", "momentum")]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in ("momentum", "ridge", "gru"):
        sub = [c for c in cells if c["model"] == m]
        if not sub:
            continue
        ax.scatter([c["turnover"] for c in sub], [c["net_return_5bps"] for c in sub],
                   s=34, alpha=0.75, color=COLORS[m], label=m, edgecolor="white",
                   linewidth=0.5)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xlabel("turnover (position changes per evaluated window)")
    ax.set_ylabel("net log return after 5 bps/side")
    ax.set_title("Costs eat the signal: net P&L vs turnover", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = ASSETS / "fig_net_vs_turnover.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_sentiment(sent: dict) -> None:
    paired = [p for p in sent["paired"] if p["comparison"].startswith("gru_sent")]
    if not paired:
        print("no paired sentiment rows; skipping")
        return
    tickers = sorted({p["ticker"] for p in paired})
    horizons = sorted({p["horizon"] for p in paired})
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    width = 0.8 / max(len(tickers), 1)
    x = np.arange(len(horizons))
    for i, t in enumerate(tickers):
        vals, sig = [], []
        for h in horizons:
            row = next((p for p in paired if p["ticker"] == t and p["horizon"] == h), None)
            vals.append(row["delta_dir_acc"] if row else 0.0)
            sig.append(bool(row and row["p_holm_sent_vs_price"] < 0.05))
        ax.bar(x + i * width, vals, width=width * 0.92, label=t, alpha=0.85)
        for xi, v, s in zip(x + i * width, vals, sig):
            ax.text(xi, v, "*" if s else "", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=12)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x + width * (len(tickers) - 1) / 2)
    ax.set_xticklabels([f"h = {h}" for h in horizons])
    ax.set_ylabel("Δ directional accuracy\n(GRU+sentiment − GRU price-only)")
    ax.set_title("Sentiment block: paired effect (bars), "
                 "* = Holm p < 0.05", fontsize=11)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_ylim(min(0.0, min(ax.get_ylim()[0], -0.12)) * 1.25,
                max(ax.get_ylim()[1], 0.12) * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    out = ASSETS / "fig_sentiment_delta.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    dec = load(DEC)
    fig_excess(dec)
    fig_net(dec)
    if SENT.exists():
        fig_sentiment(load(SENT))

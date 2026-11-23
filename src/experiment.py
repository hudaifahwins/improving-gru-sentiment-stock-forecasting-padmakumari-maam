"""Experiment harness for improving GRU sentiment stock forecasting.

Builds on :mod:`src.baseline` and adds two concrete test patterns:

* ``--mode walk-forward`` -- expanding or rolling walk-forward evaluation
  across sequential folds. Each fold fits its own scaler on that fold's
  training rows only and reports per-fold metrics plus mean/std across folds
  for RMSE, MAPE, and directional accuracy.
* ``--mode sentiment-ablation`` -- price-only vs. date-aligned sentiment
  features on the *exact same* chronological split, model config, and seed.

All runs are seeded deterministically (``random``, ``numpy``, ``torch``) and
write JSON results into ``results/``.

Usage (from the repository root)::

    python -m src.experiment --mode walk-forward --sample-synthetic
    python -m src.experiment --mode walk-forward --data data/prices.csv \
        --wf-type rolling --n-folds 5
    python -m src.experiment --mode sentiment-ablation \
        --data data/prices.csv --sentiment-cols Sentiment
    python -m src.experiment --help
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Allow `python src/experiment.py` as well as `python -m src.experiment`.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import (  # noqa: E402
    DataConfig,
    ModelConfig,
    RunConfig,
    build_model,
    evaluate_model,
    evaluate_persistence,
    make_run_id,
    prepare_dataframe,
    prepare_with_ranges,
    resolve_data_source,
    resolve_sentiment_columns,
    save_metrics_json,
    set_seed,
    train_network,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LOOKBACK,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Settings shared by walk-forward and ablation experiments."""

    mode: str = "walk-forward"  # "walk-forward" | "sentiment-ablation"
    wf_type: str = "expanding"  # "expanding" | "rolling"
    n_folds: int = 5
    rolling_window: int = 0  # 0 -> auto (3x fold size)
    lookback: int = DEFAULT_LOOKBACK
    val_frac: float = DEFAULT_VAL_FRAC
    test_frac: float = DEFAULT_TEST_FRAC
    scaler: str = "minmax"
    out_dir: Path = Path("results")


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: Any,
    *,
    data_cfg: DataConfig,
    run_cfg: ExperimentConfig,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Expanding/rolling walk-forward evaluation across sequential folds.

    For every fold the *only* information allowed to shape the model is the
    fold's training block: the scaler is fitted on training rows, windows are
    built within the block, and an inner chronological validation slice (the
    tail of the training block) drives early stopping. The held-out segment is
    then scored. No future rows ever influence a prediction.
    """
    set_seed(model_cfg.seed)
    n = len(df)
    fold_size = max(run_cfg.lookback + 2, n // (run_cfg.n_folds + 1))

    folds: list[dict[str, Any]] = []
    for k in range(run_cfg.n_folds):
        test_start = (k + 1) * fold_size
        test_end = min(n, (k + 2) * fold_size)
        if test_start >= n:
            break

        train_end = test_start
        if run_cfg.wf_type == "rolling":
            win = run_cfg.rolling_window if run_cfg.rolling_window > 0 else 3 * fold_size
            train_start = max(0, train_end - win)
        else:  # expanding
            train_start = 0

        inner_len = train_end - train_start
        if inner_len <= run_cfg.lookback + 1:
            print(f"[walk-forward] skipping fold {k}: training span too short "
                  f"({inner_len} rows)")
            continue

        # Inner validation = chronological tail of the training block.
        val_len = max(1, round(0.15 * inner_len))
        if inner_len - val_len <= run_cfg.lookback + 1:
            val_len = 0
        if val_len == 0:
            print(f"[walk-forward] skipping fold {k}: no room for an inner "
                  f"validation slice (inner_len={inner_len}, lookback={run_cfg.lookback})")
            continue
        val_start = train_end - val_len

        try:
            prep = prepare_with_ranges(
                df,
                data_cfg=data_cfg,
                lookback=run_cfg.lookback,
                horizon=1,
                train_range=(train_start, val_start),
                val_range=(val_start, train_end),
                test_range=(test_start, test_end),
                scaler_kind=run_cfg.scaler,
            )
        except ValueError as exc:
            print(f"[walk-forward] skipping fold {k}: {exc}")
            continue

        # Same seed per fold => identical model init across folds.
        set_seed(model_cfg.seed)
        model = build_model(model_cfg.model, prep.input_size, model_cfg)
        history = train_network(
            model, prep.train.X, prep.train.y_scaled, prep.val.X, prep.val.y_scaled,
            model_cfg,
        )
        ev = evaluate_model(model, prep, model_cfg)
        per = evaluate_persistence(prep)

        folds.append(
            {
                "fold": k,
                "wf_type": run_cfg.wf_type,
                "train_range": [train_start, train_end],
                "val_range": [val_start, train_end],
                "test_range": [test_start, test_end],
                "metrics": ev["metrics"],
                "persistence_metrics": per["metrics"],
                "train_time_s": history["train_time_s"],
                "infer_time_s": ev["infer_time_s"],
                "best_epoch": history["best_epoch"],
                "epochs_run": history["epochs_run"],
                "best_val_loss": history["best_val_loss"],
            }
        )

    if not folds:
        raise SystemExit("No valid folds could be built; check dataset size / params.")

    aggregate: dict[str, Any] = {}
    for key in ("rmse", "mape", "directional_accuracy"):
        model_vals = np.array([f["metrics"][key] for f in folds], dtype=float)
        pers_vals = np.array(
            [f["persistence_metrics"][key] for f in folds], dtype=float
        )
        aggregate[key] = {
            "mean": float(np.nanmean(model_vals)),
            "std": float(np.nanstd(model_vals)),
            "min": float(np.nanmin(model_vals)),
            "max": float(np.nanmax(model_vals)),
        }
        aggregate[f"persistence_{key}"] = {
            "mean": float(np.nanmean(pers_vals)),
            "std": float(np.nanstd(pers_vals)),
            "min": float(np.nanmin(pers_vals)),
            "max": float(np.nanmax(pers_vals)),
        }

    return {
        "mode": "walk-forward",
        "wf_type": run_cfg.wf_type,
        "n_folds": len(folds),
        "fold_size": fold_size,
        "folds": folds,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Sentiment ablation
# ---------------------------------------------------------------------------

def run_sentiment_ablation(
    df: Any,
    *,
    data_cfg: DataConfig,
    run_cfg: ExperimentConfig,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Compare price-only vs. price+sentiment on the identical split & seed."""
    if not data_cfg.sentiment_cols:
        raise SystemExit(
            "sentiment-ablation requires sentiment features: pass "
            "--sentiment-cols or rely on auto-detection."
        )

    price_only_cfg = replace(data_cfg, sentiment_cols=())
    variants: dict[str, Any] = {}

    for label, dc in (("price_only", price_only_cfg), ("with_sentiment", data_cfg)):
        prep = prepare_dataframe(
            df,
            data_cfg=dc,
            lookback=run_cfg.lookback,
            horizon=1,
            val_frac=run_cfg.val_frac,
            test_frac=run_cfg.test_frac,
            scaler_kind=run_cfg.scaler,
        )
        set_seed(model_cfg.seed)  # identical init/training for both variants
        model = build_model(model_cfg.model, prep.input_size, model_cfg)
        history = train_network(
            model, prep.train.X, prep.train.y_scaled, prep.val.X, prep.val.y_scaled,
            model_cfg,
        )
        ev = evaluate_model(model, prep, model_cfg)
        per = evaluate_persistence(prep)

        variants[label] = {
            "features": list(prep.feature_names),
            "input_size": prep.input_size,
            "metrics": ev["metrics"],
            "persistence_metrics": per["metrics"],
            "train_time_s": history["train_time_s"],
            "infer_time_s": ev["infer_time_s"],
            "best_epoch": history["best_epoch"],
            "epochs_run": history["epochs_run"],
            "best_val_loss": history["best_val_loss"],
        }

    delta = {
        key: variants["with_sentiment"]["metrics"][key]
        - variants["price_only"]["metrics"][key]
        for key in variants["price_only"]["metrics"]
    }

    return {
        "mode": "sentiment-ablation",
        "variants": variants,
        "delta_with_sentiment_minus_price_only": delta,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    df: Any,
    *,
    data_cfg: DataConfig,
    exp_cfg: ExperimentConfig,
    model_cfg: ModelConfig,
    source_label: str,
) -> dict[str, Any]:
    """Dispatch to the requested experiment mode and persist results."""
    set_seed(model_cfg.seed)

    if exp_cfg.mode == "walk-forward":
        results = run_walk_forward(
            df, data_cfg=data_cfg, run_cfg=exp_cfg, model_cfg=model_cfg
        )
    elif exp_cfg.mode == "sentiment-ablation":
        results = run_sentiment_ablation(
            df, data_cfg=data_cfg, run_cfg=exp_cfg, model_cfg=model_cfg
        )
    else:  # pragma: no cover - argparse restricts choices
        raise SystemExit(f"Unknown mode: {exp_cfg.mode!r}")

    run_id = make_run_id(f"experiment_{exp_cfg.mode}")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "project": "improving-gru-sentiment-stock-forecasting-padmakumari-maam",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": source_label,
        "n_rows": len(df),
        "config": {
            "experiment": asdict(exp_cfg),
            "model": asdict(model_cfg),
            "data": asdict(data_cfg),
        },
        "results": results,
    }
    saved = save_metrics_json(exp_cfg.out_dir, run_id, payload)
    payload["saved"] = str(saved)
    save_metrics_json(exp_cfg.out_dir, run_id, payload)

    print_experiment_summary(payload)
    return payload


def print_experiment_summary(payload: dict[str, Any]) -> None:
    results = payload["results"]
    cfg = payload["config"]
    print("\n" + "=" * 72)
    print(f"EXPERIMENT SUMMARY - {results['mode'].upper()}")
    print("=" * 72)
    print(f"run_id      : {payload['run_id']}")
    print(f"data_source : {payload['data_source']}  ({payload['n_rows']} rows)")
    print(f"model       : {cfg['model']['model']} (hidden={cfg['model']['hidden_size']}, "
          f"seed={cfg['model']['seed']})")
    print(f"lookback    : {cfg['experiment']['lookback']}")

    if results["mode"] == "walk-forward":
        print(f"wf_type     : {results['wf_type']}  folds={results['n_folds']}")
        print("\n-- Aggregate over folds (mean ± std) --")
        agg = results["aggregate"]
        for key in ("rmse", "mape", "directional_accuracy"):
            print(f"   {key:22s} model={agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}  "
                  f"persistence={agg['persistence_'+key]['mean']:.4f} ± "
                  f"{agg['persistence_'+key]['std']:.4f}")
        print("\n-- Per-fold metrics --")
        for f in results["folds"]:
            m = f["metrics"]
            print(f"   fold {f['fold']}: test=[{f['test_range'][0]},{f['test_range'][1]})  "
                  f"RMSE={m['rmse']:.4f} MAPE={m['mape']:.3f}% "
                  f"DirAcc={m['directional_accuracy']:.4f}")

    elif results["mode"] == "sentiment-ablation":
        print("\n-- Variants (same split & seed) --")
        for label, variant in results["variants"].items():
            m = variant["metrics"]
            print(f"   {label:16s} features={variant['features']}")
            print(f"      MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                  f"MAPE={m['mape']:.3f}% R2={m['r2']:.4f} "
                  f"DirAcc={m['directional_accuracy']:.4f}")
        print("\n-- Delta (with_sentiment - price_only) --")
        for key, value in results["delta_with_sentiment_minus_price_only"].items():
            print(f"   {key:22s} {value:+.4f}")

    if payload.get("saved"):
        print(f"\nsaved: {payload['saved']}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description=(
            "Experiments on top of the baseline: walk-forward evaluation and "
            "sentiment ablation, both leakage-free and seeded deterministically."
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.experiment --mode walk-forward --sample-synthetic\n"
            "  python -m src.experiment --mode walk-forward --data data/prices.csv \\\n"
            "      --wf-type rolling --n-folds 5\n"
            "  python -m src.experiment --mode sentiment-ablation \\\n"
            "      --data data/prices.csv --sentiment-cols Sentiment\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["walk-forward", "sentiment-ablation"],
        default="walk-forward",
        help="which experiment to run (default: walk-forward)",
    )

    wf = parser.add_argument_group("walk-forward settings")
    wf.add_argument("--wf-type", choices=["expanding", "rolling"], default="expanding")
    wf.add_argument("--n-folds", type=int, default=5)
    wf.add_argument("--wf-train-window", type=int, default=0,
                    help="rolling train window length (0 = auto: 3x fold size)")

    data = parser.add_argument_group("data source (exactly one required)")
    data.add_argument("--data", metavar="FILE", help="path to a price CSV")
    data.add_argument("--ticker", metavar="SYMBOL", help="Yahoo Finance ticker (yfinance)")
    data.add_argument("--sample-synthetic", action="store_true",
                      help="generate a small synthetic series for testing")
    data.add_argument("--date-col", default="Date")
    data.add_argument("--close-col", default="Close")
    data.add_argument("--sentiment-cols", default="", metavar="COLS",
                      help="comma-separated sentiment columns (default: auto-detect)")
    data.add_argument("--no-sentiment", action="store_true",
                      help="disable sentiment features (price-only)")

    win = parser.add_argument_group("windowing / splitting")
    win.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    win.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    win.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    win.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")

    mod = parser.add_argument_group("model / training")
    mod.add_argument("--model", choices=["gru", "lstm"], default="gru")
    mod.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    mod.add_argument("--layers", type=int, default=None)
    mod.add_argument("--dropout", type=float, default=0.0)
    mod.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    mod.add_argument("--lr", type=float, default=DEFAULT_LR)
    mod.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    mod.add_argument("--batch-size", type=int, default=64)
    mod.add_argument("--seed", type=int, default=DEFAULT_SEED)
    mod.add_argument("--device", default="auto", help="auto | cpu | cuda")

    out = parser.add_argument_group("output")
    out.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.no_sentiment:
        sentiment_cols: tuple[str, ...] = ()
    else:
        sentiment_cols = tuple(
            c.strip() for c in args.sentiment_cols.split(",") if c.strip()
        )

    df, source_label = resolve_data_source(
        path=args.data,
        ticker=args.ticker,
        synthetic=args.sample_synthetic,
        seed=args.seed,
        date_col=args.date_col,
        close_col=args.close_col,
        sentiment_cols=sentiment_cols,
    )

    if not args.no_sentiment and not sentiment_cols:
        sentiment_cols = resolve_sentiment_columns(
            df, date_col=args.date_col, close_col=args.close_col
        )
        if sentiment_cols:
            print(f"[data] auto-detected sentiment column(s): {list(sentiment_cols)}")

    if sentiment_cols:
        missing = [c for c in sentiment_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Sentiment column(s) not found in data: {missing}")

    data_cfg = DataConfig(
        date_col=args.date_col,
        close_col=args.close_col,
        sentiment_cols=sentiment_cols,
    )
    model_cfg = ModelConfig(
        seed=args.seed,
        model=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    exp_cfg = ExperimentConfig(
        mode=args.mode,
        wf_type=args.wf_type,
        n_folds=args.n_folds,
        rolling_window=args.wf_train_window,
        lookback=args.lookback,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        scaler=args.scaler,
        out_dir=args.out_dir,
    )

    try:
        run_experiment(
            df,
            data_cfg=data_cfg,
            exp_cfg=exp_cfg,
            model_cfg=model_cfg,
            source_label=source_label,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()

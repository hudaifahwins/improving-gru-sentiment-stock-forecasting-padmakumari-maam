"""Stress tests for the baseline GRU harness.

Runs three controlled checks on the *exact same* chronological split and
deterministic seed, reusing the data loading, windowing, training, and metric
helpers from :mod:`src.baseline`:

* Check 1 - Scaler Leakage Test: MinMaxScaler fit on the full series vs. fit
  strictly on the training split; logs the metric delta.
* Check 2 - Persistence vs. GRU Test: naive persistence (``y_hat_t = y_{t-1}``)
  vs. :class:`BaselineGRU`, making raw price-level error vs. directional
  accuracy explicit.
* Check 3 - Sentiment / Feature Sanity Check: price-only vs. date-aligned
  sentiment vs. time-shuffled sentiment on the same split. Skipped cleanly
  (``status: "skipped_no_sentiment_cols"``) when no sentiment columns exist.

All metrics include ``nrmse`` (``rmse / (y_max - y_min)`` on unscaled target
prices, NaN on a zero range) alongside MAE, RMSE, MAPE, R², and directional
accuracy. A structured JSON report and a readable Markdown summary are written
under ``results/``.

Usage (from the repository root)::

    python -m src.stress_test --help
    python -m src.stress_test --sample-synthetic --seed 42
    python -m src.stress_test --ticker AAPL --seed 42
    python -m src.stress_test --csv data/prices.csv --sentiment-cols Sentiment
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow `python src/stress_test.py` as well as `python -m src.stress_test`.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import (  # noqa: E402
    DataConfig,
    ModelConfig,
    build_model,
    evaluate_model,
    evaluate_persistence,
    fit_scaler,
    make_run_id,
    prepare_with_ranges,
    require_torch,
    resolve_data_source,
    resolve_sentiment_columns,
    save_metrics_json,
    set_seed,
    train_network,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_LOOKBACK,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
)

METRIC_KEYS = ("mae", "rmse", "nrmse", "mape", "r2", "directional_accuracy")


@dataclass
class StressConfig:
    """Settings for one stress-test run."""

    lookback: int = DEFAULT_LOOKBACK
    seed: int = DEFAULT_SEED
    out_dir: Path = Path("results")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chrono_ranges(
    n: int,
    lookback: int,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """The canonical chronological 70/15/15 split (same math as baseline)."""
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train <= lookback + 1:
        raise ValueError(
            f"Training split ({n_train} rows) is too small for lookback={lookback}."
        )
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)


def _feature_matrix(df: pd.DataFrame, data_cfg: DataConfig) -> np.ndarray:
    features = [data_cfg.close_col, *data_cfg.sentiment_cols]
    return df[features].to_numpy(dtype=float)


def _train_and_evaluate_gru(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    lookback: int,
    train_range: tuple[int, int],
    val_range: tuple[int, int],
    test_range: tuple[int, int],
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Train a GRU + persistence on one prep and evaluate the held-out test."""
    prep = prepare_with_ranges(
        df,
        data_cfg=data_cfg,
        lookback=lookback,
        horizon=1,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        scaler_kind="minmax",
        scaler=scaler,
    )
    set_seed(model_cfg.seed)
    model = build_model("gru", prep.input_size, model_cfg)
    history = train_network(
        model, prep.train.X, prep.train.y_scaled, prep.val.X, prep.val.y_scaled,
        model_cfg,
    )
    ev = evaluate_model(model, prep, model_cfg)
    per = evaluate_persistence(prep)
    return {
        "model": {
            "metrics": ev["metrics"],
            "train_time_s": history["train_time_s"],
            "infer_time_s": ev["infer_time_s"],
            "best_epoch": history["best_epoch"],
            "epochs_run": history["epochs_run"],
            "best_val_loss": history["best_val_loss"],
        },
        "persistence": {
            "metrics": per["metrics"],
            "infer_time_s": per["infer_time_s"],
        },
    }


def _fmt_metrics(metrics: dict[str, float]) -> str:
    return (
        f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} "
        f"NRMSE={metrics['nrmse']:.4f} MAPE={metrics['mape']:.3f}% "
        f"R2={metrics['r2']:.4f} DirAcc={metrics['directional_accuracy']:.4f}"
    )


def _metric_delta(base: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    return {k: other[k] - base[k] for k in METRIC_KEYS}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check1_scaler_leakage(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    lookback: int,
    ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> dict[str, Any]:
    """MinMaxScaler fitted on the full series vs. on the training split only."""
    train_range, val_range, test_range = ranges
    full_m = _feature_matrix(df, data_cfg)
    leaky_scaler = fit_scaler(full_m, "minmax")

    leaky = _train_and_evaluate_gru(
        df, data_cfg=data_cfg, model_cfg=model_cfg, lookback=lookback,
        train_range=train_range, val_range=val_range, test_range=test_range,
        scaler=leaky_scaler,
    )
    clean = _train_and_evaluate_gru(
        df, data_cfg=data_cfg, model_cfg=model_cfg, lookback=lookback,
        train_range=train_range, val_range=val_range, test_range=test_range,
    )

    return {
        "description": (
            "MinMaxScaler fit on the full series before splitting (mild leakage) "
            "vs. fit strictly on the training split; same windows, model config, "
            "and seed."
        ),
        "scaler_fit_on_full_series": leaky["model"],
        "scaler_fit_on_train_only": clean["model"],
        "delta_full_fit_minus_train_only": _metric_delta(
            clean["model"]["metrics"], leaky["model"]["metrics"]
        ),
    }


def check2_persistence_vs_gru(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persistence vs. GRU on the canonical train-only prep."""
    return {
        "description": (
            "Naive persistence (y_hat_t = y_{t-1}) vs. BaselineGRU on the same "
            "chronological split; raw price-level error and directional accuracy "
            "are reported side by side."
        ),
        "persistence": result["persistence"],
        "gru": result["model"],
        "takeaway": (
            "Persistence scores lowest on raw price-level error (MAE/RMSE/MAPE); "
            "directional accuracy is the metric where a learned model can add "
            "value over persistence."
        ),
    }


def check3_sentiment_sanity(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    lookback: int,
    ranges: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> dict[str, Any]:
    """Price-only vs. date-aligned sentiment vs. time-shuffled sentiment."""
    if not data_cfg.sentiment_cols:
        return {"status": "skipped_no_sentiment_cols"}

    train_range, val_range, test_range = ranges

    price_only_cfg = DataConfig(
        date_col=data_cfg.date_col,
        close_col=data_cfg.close_col,
        sentiment_cols=(),
    )

    # Time-shuffled: permute each sentiment column across rows (breaks temporal
    # alignment, keeps the same marginal distribution).
    rng = np.random.default_rng(model_cfg.seed + 1234)
    shuffled_df = df.copy()
    for col in data_cfg.sentiment_cols:
        values = df[col].to_numpy()
        shuffled_df[col] = values[rng.permutation(len(values))]

    variants: dict[str, Any] = {}
    for label, frame, cfg in (
        ("price_only", df, price_only_cfg),
        ("sentiment_aligned", df, data_cfg),
        ("sentiment_shuffled", shuffled_df, data_cfg),
    ):
        variants[label] = _train_and_evaluate_gru(
            frame, data_cfg=cfg, model_cfg=model_cfg, lookback=lookback,
            train_range=train_range, val_range=val_range, test_range=test_range,
        )["model"]

    return {
        "status": "completed",
        "description": (
            "Price-only vs. date-aligned sentiment vs. time-shuffled sentiment "
            "features on the exact same split, model config, and seed."
        ),
        "sentiment_columns": list(data_cfg.sentiment_cols),
        "variants": variants,
        "delta_vs_price_only": {
            label: _metric_delta(
                variants["price_only"]["metrics"], variants[label]["metrics"]
            )
            for label in ("sentiment_aligned", "sentiment_shuffled")
        },
    }


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------

def run_stress_tests(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    stress_cfg: StressConfig,
    source_label: str,
) -> dict[str, Any]:
    """Execute all three checks and persist JSON + Markdown reports."""
    require_torch()
    set_seed(model_cfg.seed)

    n = len(df)
    ranges = _chrono_ranges(n, stress_cfg.lookback)

    # Canonical train-only run reused by checks 2 (and 1's clean variant).
    canonical = _train_and_evaluate_gru(
        df,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        lookback=stress_cfg.lookback,
        train_range=ranges[0],
        val_range=ranges[1],
        test_range=ranges[2],
    )

    results = {
        "check_1_scaler_leakage": check1_scaler_leakage(
            df, data_cfg=data_cfg, model_cfg=model_cfg,
            lookback=stress_cfg.lookback, ranges=ranges
        ),
        "check_2_persistence_vs_gru": check2_persistence_vs_gru(canonical),
        "check_3_sentiment_sanity": check3_sentiment_sanity(
            df, data_cfg=data_cfg, model_cfg=model_cfg,
            lookback=stress_cfg.lookback, ranges=ranges
        ),
    }

    run_id = make_run_id("stress_test")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "project": "improving-gru-sentiment-stock-forecasting-padmakumari-maam",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": source_label,
        "n_rows": n,
        "config": {
            "stress": asdict(stress_cfg),
            "model": asdict(model_cfg),
            "data": asdict(data_cfg),
        },
        "split": {
            "train_range": list(ranges[0]),
            "val_range": list(ranges[1]),
            "test_range": list(ranges[2]),
        },
        "metric_keys": list(METRIC_KEYS),
        "results": results,
    }

    json_path = save_metrics_json(stress_cfg.out_dir, run_id, payload)
    md_path = _write_markdown(stress_cfg.out_dir, run_id, payload)
    payload["saved"] = {"json": str(json_path), "markdown": str(md_path)}
    save_metrics_json(stress_cfg.out_dir, run_id, payload)

    _print_summary(payload)
    return payload


def _write_markdown(
    out_dir: Path, run_id: str, payload: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    cfg = payload["config"]
    lines = [
        f"# Stress test report — `{run_id}`",
        "",
        f"- **Data source:** {payload['data_source']} ({payload['n_rows']} rows)",
        f"- **Split:** chronological {cfg['stress']['lookback']}-based windows; "
        f"train={payload['split']['train_range']}, val={payload['split']['val_range']}, "
        f"test={payload['split']['test_range']}",
        f"- **Model:** GRU (hidden={cfg['model']['hidden_size']}, seed={cfg['model']['seed']})",
        f"- **Metrics:** {', '.join(payload['metric_keys'])}",
        "",
    ]

    results = payload["results"]

    lines += ["## Check 1 — Scaler leakage test", ""]
    c1 = results["check_1_scaler_leakage"]
    lines += [c1["description"], ""]
    for label, key in (
        ("Scaler fit on full series", "scaler_fit_on_full_series"),
        ("Scaler fit on train only", "scaler_fit_on_train_only"),
    ):
        m = c1[key]["metrics"]
        lines += [
            f"- **{label}:** {_fmt_metrics(m)} "
            f"(best_epoch={c1[key]['best_epoch']}, epochs_run={c1[key]['epochs_run']})"
        ]
    delta = c1["delta_full_fit_minus_train_only"]
    lines += ["", "Delta (full fit − train only):"]
    for k in METRIC_KEYS:
        lines.append(f"  - `{k}`: {delta[k]:+.4f}")
    lines += [""]

    lines += ["## Check 2 — Persistence vs. GRU", ""]
    c2 = results["check_2_persistence_vs_gru"]
    lines += [c2["description"], ""]
    lines += [
        "| model | MAE | RMSE | NRMSE | MAPE | R2 | DirAcc |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in (("persistence", "persistence"), ("GRU", "gru")):
        m = c2[key]["metrics"]
        lines.append(
            f"| {label} | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{m['nrmse']:.4f} | {m['mape']:.3f}% | {m['r2']:.4f} | "
            f"{m['directional_accuracy']:.4f} |"
        )
    lines += ["", c2["takeaway"], ""]

    lines += ["## Check 3 — Sentiment / feature sanity", ""]
    c3 = results["check_3_sentiment_sanity"]
    if c3.get("status") == "skipped_no_sentiment_cols":
        lines += ["`status: skipped_no_sentiment_cols` — no sentiment columns detected.", ""]
    else:
        lines += [c3["description"], ""]
        lines += [
            "| variant | MAE | RMSE | NRMSE | MAPE | R2 | DirAcc |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, variant in c3["variants"].items():
            m = variant["metrics"]
            lines.append(
                f"| {label} | {m['mae']:.4f} | {m['rmse']:.4f} | "
                f"{m['nrmse']:.4f} | {m['mape']:.3f}% | {m['r2']:.4f} | "
                f"{m['directional_accuracy']:.4f} |"
            )
        lines += ["", "Delta vs. price-only:"]
        for label, delta in c3["delta_vs_price_only"].items():
            lines.append(f"  - `{label}`: " + ", ".join(
                f"{k}={v:+.4f}" for k, v in delta.items()
            ))
        lines += [""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_summary(payload: dict[str, Any]) -> None:
    cfg = payload["config"]
    results = payload["results"]
    print("\n" + "=" * 72)
    print("STRESS TEST SUMMARY")
    print("=" * 72)
    print(f"run_id      : {payload['run_id']}")
    print(f"data_source : {payload['data_source']}  ({payload['n_rows']} rows)")
    print(f"seed        : {cfg['model']['seed']}  lookback={cfg['stress']['lookback']}")

    c1 = results["check_1_scaler_leakage"]
    print("\n[1] Scaler leakage test")
    print("    full fit : " + _fmt_metrics(c1["scaler_fit_on_full_series"]["metrics"]))
    print("    train-fit: " + _fmt_metrics(c1["scaler_fit_on_train_only"]["metrics"]))
    d1 = c1["delta_full_fit_minus_train_only"]
    print("    delta (full − train): "
          + ", ".join(f"{k}={v:+.4f}" for k, v in d1.items()))

    c2 = results["check_2_persistence_vs_gru"]
    print("\n[2] Persistence vs. GRU")
    print("    persistence: " + _fmt_metrics(c2["persistence"]["metrics"]))
    print("    gru       : " + _fmt_metrics(c2["gru"]["metrics"]))

    c3 = results["check_3_sentiment_sanity"]
    print("\n[3] Sentiment sanity check")
    if c3.get("status") == "skipped_no_sentiment_cols":
        print("    status: skipped_no_sentiment_cols")
    else:
        for label, variant in c3["variants"].items():
            print(f"    {label:20s}: " + _fmt_metrics(variant["metrics"]))

    print("\nsaved:")
    for key, val in payload["saved"].items():
        print(f"   {key}: {val}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.stress_test",
        description=(
            "Stress tests for the baseline GRU harness: scaler-leakage, "
            "persistence-vs-GRU, and sentiment-feature sanity checks on one "
            "chronological split with a deterministic seed."
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.stress_test --sample-synthetic --seed 42\n"
            "  python -m src.stress_test --ticker AAPL --seed 42\n"
            "  python -m src.stress_test --csv data/prices.csv --sentiment-cols Sentiment\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = parser.add_argument_group("data source (exactly one required)")
    data.add_argument("--csv", metavar="FILE", help="path to a price CSV")
    data.add_argument("--ticker", metavar="SYMBOL", help="Yahoo Finance ticker (yfinance)")
    data.add_argument("--sample-synthetic", action="store_true",
                      help="generate a small synthetic series for testing")
    data.add_argument("--date-col", default="Date")
    data.add_argument("--close-col", default="Close")
    data.add_argument("--sentiment-cols", default="", metavar="COLS",
                      help="comma-separated sentiment columns (default: auto-detect)")
    data.add_argument("--no-sentiment", action="store_true",
                      help="disable sentiment features entirely (price-only)")

    run = parser.add_argument_group("stress test settings")
    run.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    run.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    run.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    run.add_argument("--lr", type=float, default=DEFAULT_LR)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--device", default="auto", help="auto | cpu | cuda")
    run.add_argument("--out-dir", type=Path, default=Path("results"))
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
        path=args.csv,
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
        model="gru",
        hidden_size=args.hidden_size,
        num_layers=None,
        dropout=0.0,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    stress_cfg = StressConfig(
        lookback=args.lookback,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    try:
        run_stress_tests(
            df,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            stress_cfg=stress_cfg,
            source_label=source_label,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()

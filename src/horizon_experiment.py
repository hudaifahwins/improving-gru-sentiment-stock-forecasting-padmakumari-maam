"""Horizon / feature / model-family diagnostic experiments (leak-free).

Evaluates next-return forecasting for horizons h ∈ {1, 3, 5} with:
* price-only features, causal technical features (rolling returns/vol/momentum/
  range, all past-only), and real GDELT-derived sentiment transforms (rolling
  tone, tone shock, active-news-day count),
* GRU (mse / hybrid loss) vs. Ridge vs. HistGradientBoostingRegressor
  baselines on the identical chronological 70/15/15 split per seed.

Discipline: scaler fit on train rows only; windows are built per split segment
and targets newer require ``t - h`` to lie inside the segment (no cross-split
look-ahead); sentiment is already lagged (pre-close) by the ingestion script.

Usage::
    python -m src.horizon_experiment --ticker AAPL --seed 42 --horizon 3 \
        --features price+tech --loss mse --baseline ridge --epochs 100
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import (  # noqa: E402
    DataConfig,
    ModelConfig,
    fit_scaler,
    compute_metrics,
    make_run_id,
    resolve_data_source,
    resolve_sentiment_columns,
    save_metrics_json,
    set_seed,
    DEFAULT_LOOKBACK,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
)
from src.return_experiment import (  # noqa: E402
    HybridGRU,
    _train_return_model,
    _predict_returns,
    _reconstruct_prices,
    compute_return_metrics,
)

DEFAULT_EPOCHS = 200


@dataclass
class HConfig:
    lookback: int = DEFAULT_LOOKBACK
    horizon: int = 1
    val_frac: float = DEFAULT_VAL_FRAC
    test_frac: float = DEFAULT_TEST_FRAC
    loss: str = "mse"
    return_type: str = "log"
    deadzone_eps: float = 0.005
    baseline: str = "none"  # none | ridge | gbr
    features: str = "price"  # price | price+tech | price+tech+sentiment
    seed: int = DEFAULT_SEED
    out_dir: Path = Path("results")


# ---------------------------------------------------------------------------
# causal features
# ---------------------------------------------------------------------------

def build_feature_frame(df: pd.DataFrame, data_cfg: DataConfig,
                        features: str) -> pd.DataFrame:
    """Return a DataFrame of causal (past-only) features, aligned to rows.

    All rolling/momentum ops use only information available at/close of each
    row. Sentiment input is already lagged by the ingestion pipeline.
    """
    close = df[data_cfg.close_col].astype(float)
    log_close = np.log(close)
    ret = log_close.diff()
    out = pd.DataFrame(index=df.index)
    out["Close"] = close  # called feature (scaled) and source of target prices
    if "tech" in features:
        out["ret_mean5"] = ret.rolling(5, min_periods=1).mean()
        out["ret_std10"] = ret.rolling(10, min_periods=1).std()
        out["mom20"] = log_close - log_close.shift(20)
        out["range10"] = (close.rolling(10).max() - close.rolling(10).min()) / close.shift(1)
        out["log_close"] = log_close
    if "sentiment" in features and data_cfg.sentiment_cols:
        tone = df[data_cfg.sentiment_cols[0]].astype(float)
        out["tone5"] = tone.rolling(5, min_periods=1).mean()
        out["tone_shock"] = tone.diff()
        out["active10"] = (tone != 0).astype(float).rolling(10, min_periods=1).sum()
    return out.dropna()


# ---------------------------------------------------------------------------
# leak-free windowing for horizon h
# ---------------------------------------------------------------------------

def build_h_windows(feat: np.ndarray, close: np.ndarray, lookback: int,
                    horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows where target t uses rows [t-lb, t) and P_{t-h}.

    All inputs here are already restricted to a single split segment, so targets
    that would reference rows before the segment (t-h < segment start) are
    excluded.
    """
    n = len(feat)
    idx = np.arange(n)
    # target rows t with t >= lookback and t - horizon >= 0 within segment
    keep = (idx >= lookback) & ((idx - horizon) >= 0)
    t_idx = np.flatnonzero(keep)
    if len(t_idx) == 0:
        raise ValueError("No valid window rows for this segment size/horizon.")
    # X windows: for each t, [t-lb, t)
    x = np.empty((len(t_idx), lookback, feat.shape[1]), dtype=np.float32)
    for i, t in enumerate(t_idx):
        x[i] = feat[t - lookback:t]
    y = np.log(close[t_idx] / close[t_idx - horizon]).astype(np.float32)
    prev = close[t_idx - horizon]
    return x, y.reshape(-1, 1), prev


def prepare_h(df: pd.DataFrame, *, data_cfg: DataConfig, cfg: HConfig,
              features: str) -> tuple[Any, Any, Any, Any, Any]:
    """Return (scaled features, close, ranges, scaler, names)."""
    fdf = build_feature_frame(df, data_cfg, features)
    close = fdf["Close"].to_numpy(float)
    names = list(fdf.columns)  # includes "Close" as the first feature
    m = fdf[names].to_numpy(float)
    n = len(m)
    n_test = max(1, round(n * cfg.test_frac))
    n_val = max(1, round(n * cfg.val_frac))
    n_train = n - n_val - n_test
    min_seg = cfg.lookback + cfg.horizon + 1
    if min(n_train, n_val, n_test) < min_seg:
        raise ValueError(
            f"Segment too small (train={n_train}, val={n_val}, test={n_test}) "
            f"for lookback={cfg.lookback}, horizon={cfg.horizon}."
        )
    scaler = fit_scaler(m[:n_train], "minmax")
    scaled = scaler.transform(m)
    tr = build_h_windows(scaled[:n_train], close[:n_train], cfg.lookback, cfg.horizon)
    va = build_h_windows(scaled[n_train:n_train + n_val], close[n_train:n_train + n_val],
                          cfg.lookback, cfg.horizon)
    te = build_h_windows(scaled[n_train + n_val:], close[n_train + n_val:],
                         cfg.lookback, cfg.horizon)
    ranges = ((0, n_train), (n_train, n_train + n_val), (n_train + n_val, n))
    return tr, va, te, scaler, (ranges, names)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def run_one(df: pd.DataFrame, *, data_cfg: DataConfig, cfg: HConfig,
            model_cfg: ModelConfig, source_label: str) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    set_seed(cfg.seed)
    tr, va, te, scaler, (ranges, names) = prepare_h(
        df, data_cfg=data_cfg, cfg=cfg, features=cfg.features
    )
    x_tr, y_tr, prev_tr = tr
    x_va, y_va, _ = va
    x_te, y_te, prev_te = te

    hybrid = cfg.loss == "hybrid"
    if hybrid:
        model = HybridGRU(x_tr.shape[2], hidden_size=model_cfg.hidden_size,
                          num_layers=model_cfg.num_layers or 1)
    else:
        from src.baseline import build_model
        model = build_model("gru", x_tr.shape[2], model_cfg)
    hist = _train_return_model(model, x_tr, y_tr, x_va, y_va, model_cfg,
                               loss_mode=cfg.loss)
    t0 = time.perf_counter()
    pred = _predict_returns(model, x_te, batch_size=model_cfg.batch_size,
                            device=model_cfg.device, hybrid=hybrid)
    infer = time.perf_counter() - t0
    pred = pred * (1.0 if not cfg.return_type.endswith("vol") else 1.0)

    out: dict[str, Any] = {}
    price = _reconstruct_prices(prev_te, pred, "log")
    out["gru"] = {
        "price_metrics": compute_metrics(np.exp(np.log(prev_te) + y_te[:, 0]),
                                         price, prev_te),
        "return_metrics": compute_return_metrics(
            y_te[:, 0], pred[:, 0], "log", deadzone_eps=cfg.deadzone_eps
        ),
        "best_epoch": hist["best_epoch"],
        "epochs_run": hist["epochs_run"],
        "train_time_s": hist["train_time_s"],
        "infer_time_s": infer,
    }

    # baselines (same windows)
    flat_tr = x_tr.reshape(x_tr.shape[0], -1)
    flat_te = x_te.reshape(x_te.shape[0], -1)
    for bname, bl in (("ridge", Ridge(alpha=1.0)),
                      ("gbr", HistGradientBoostingRegressor(random_state=cfg.seed,
                                                            max_iter=300))):
        if cfg.baseline == bname:
            bl.fit(flat_tr, y_tr.ravel())
            bp = bl.predict(flat_te).reshape(-1, 1)
            bprice = _reconstruct_prices(prev_te, bp, "log")
            out[bname] = {
                "price_metrics": compute_metrics(
                    np.exp(np.log(prev_te) + y_te[:, 0]), bprice, prev_te
                ),
                "return_metrics": compute_return_metrics(
                    y_te[:, 0], bp[:, 0], "log", deadzone_eps=cfg.deadzone_eps
                ),
                "model": type(bl).__name__,
            }

    run_id = make_run_id("horizon_experiment")
    payload = {
        "run_id": run_id,
        "project": "improving-gru-sentiment-stock-forecasting-padmakumari-maam",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": source_label,
        "n_rows": len(df),
        "config": {"h": asdict(cfg), "model": asdict(model_cfg),
                   "data": asdict(data_cfg)},
        "features": names,
        "split": {k: list(v) for k, v in zip(("train", "val", "test"), ranges)},
        "results": out,
    }
    save_metrics_json(cfg.out_dir, run_id, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.horizon_experiment")
    p.add_argument("--ticker")
    p.add_argument("--csv")
    p.add_argument("--sample-synthetic", action="store_true")
    p.add_argument("--sentiment-cols", default="")
    p.add_argument("--no-sentiment", action="store_true")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--features", default="price",
                   choices=["price", "price+tech", "price+tech+sentiment"])
    p.add_argument("--loss", choices=["mse", "hybrid"], default="mse")
    p.add_argument("--baseline", choices=["none", "ridge", "gbr"], default="none")
    p.add_argument("--eps-deadzone", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden-size", type=int, default=50)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.no_sentiment:
        sentiment_cols: tuple[str, ...] = ()
    else:
        sentiment_cols = tuple(c.strip() for c in args.sentiment_cols.split(",") if c.strip())
    df, source_label = resolve_data_source(
        path=args.csv, ticker=args.ticker, synthetic=args.sample_synthetic,
        seed=args.seed, sentiment_cols=sentiment_cols,
    )
    if not args.no_sentiment and not sentiment_cols:
        sentiment_cols = resolve_sentiment_columns(df)
        if sentiment_cols:
            print(f"[data] auto-detected sentiment column(s): {list(sentiment_cols)}")
    if sentiment_cols:
        missing = [c for c in sentiment_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Sentiment column(s) not found in data: {missing}")
    data_cfg = DataConfig(sentiment_cols=sentiment_cols)
    model_cfg = ModelConfig(
        seed=args.seed, model="gru", hidden_size=args.hidden_size,
        num_layers=args.layers, epochs=args.epochs, lr=args.lr,
        patience=args.patience, batch_size=args.batch_size, device=args.device,
    )
    cfg = HConfig(
        horizon=args.horizon, loss=args.loss, baseline=args.baseline,
        deadzone_eps=args.eps_deadzone, features=args.features,
        seed=args.seed, out_dir=args.out_dir,
    )
    try:
        payload = run_one(df, data_cfg=data_cfg, cfg=cfg, model_cfg=model_cfg,
                          source_label=source_label)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    r = payload["results"]["gru"]
    print(f"[{payload['run_id']}] {source_label} h={cfg.horizon} "
          f"feat={cfg.features} loss={cfg.loss} baseline={cfg.baseline} seed={cfg.seed}")
    print(f"   GRU DirAcc={r['return_metrics']['directional_accuracy']:.4f} "
          f"strict={r['return_metrics']['directional_accuracy_strict']:.4f} "
          f"MAE={r['price_metrics']['mae']:.4f}")
    for b in ("ridge", "gbr"):
        if b in payload["results"]:
            br = payload["results"][b]["return_metrics"]
            bp = payload["results"][b]["price_metrics"]
            print(f"   {b} DirAcc={br['directional_accuracy']:.4f} strict={br['directional_accuracy_strict']:.4f} "
                  f"MAE={bp['mae']:.4f}")


if __name__ == "__main__":
    main()

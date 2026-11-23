"""Baseline forecasting models for improving GRU sentiment stock forecasting.

This module implements the *baseline* side of the project:

* Data loading and cleaning (CSV, yfinance ticker, or a synthetic example) with
  optional date-aligned sentiment columns.
* Strict time-series hygiene: chronological sorting, past-only missing-value
  handling (forward-fill), chronological train/val/test splits, and scalers
  fitted on the training split *only*.
* Sliding-window datasets (default lookback ``30``, matching the reference
  GRU/MLS-LSTM paper).
* Models (PyTorch): naive persistence, :class:`BaselineGRU`,
  :class:`BaselineLSTM`.
* Metrics in original price units (MAE, RMSE, MAPE, R2, directional accuracy)
  plus timing, saved as JSON and appended as CSV rows under ``results/``.

Usage (from the repository root)::

    python -m src.baseline --sample-synthetic
    python -m src.baseline --data data/prices.csv --sentiment-cols Sentiment
    python -m src.baseline --ticker AAPL --lookback 30 --model lstm
    python -m src.baseline --help
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:  # PyTorch is optional at import time (needed only for model work).
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment without torch
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK = 30  # matches the reference GRU/MLS-LSTM paper
DEFAULT_HIDDEN = 50
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 15
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
HORIZON = 1

# Column names treated as sentiment features when auto-detected.
SENTIMENT_COL_RE = re.compile(r"(?i)sentiment|polarity|tone|news|score")

_CSV_COLUMNS = [
    "run_id", "timestamp", "model", "data_source", "n_rows", "lookback",
    "hidden_size", "num_layers", "epochs_run", "best_epoch",
    "train_time_s", "infer_time_s",
    "mae", "rmse", "nrmse", "mape", "r2", "directional_accuracy",
    "persistence_mae", "persistence_rmse", "persistence_nrmse",
    "persistence_mape", "persistence_r2", "persistence_directional_accuracy",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """How to read and align a raw dataset."""

    date_col: str = "Date"
    close_col: str = "Close"
    sentiment_cols: tuple[str, ...] = ()


@dataclass
class ModelConfig:
    """Model / training hyper-parameters."""

    seed: int = DEFAULT_SEED
    model: str = "gru"  # "gru" | "lstm"
    hidden_size: int = DEFAULT_HIDDEN
    num_layers: int | None = None  # None -> auto (1 for GRU, 2 for LSTM)
    dropout: float = DEFAULT_DROPOUT
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass
class RunConfig:
    """Window / split / output settings for one run."""

    lookback: int = DEFAULT_LOOKBACK
    horizon: int = HORIZON
    val_frac: float = DEFAULT_VAL_FRAC
    test_frac: float = DEFAULT_TEST_FRAC
    scaler: str = "minmax"  # "minmax" | "standard"
    run_persistence: bool = True
    plot: bool = False
    out_dir: Path = Path("results")


# ---------------------------------------------------------------------------
# Seed / device helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every random source for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def require_torch() -> None:
    """Raise a clear error if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise SystemExit(
            "PyTorch is required for this operation but was not found. "
            "Install it with: pip install torch"
        )


def resolve_device(requested: str = "auto") -> str:
    require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse ``date_col`` to ``_date`` (datetime64), dropping unparseable rows."""
    out = df.copy()
    out["_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["_date"])
    return out


def sort_and_clean(
    df: pd.DataFrame,
    *,
    date_col: str,
    close_col: str,
    sentiment_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Sort chronologically and handle missing values using only past rows.

    Missing values are forward-filled (``ffill``), which uses *earlier* rows
    only -- never future rows -- so there is no look-ahead leakage. Leading
    NaNs (which cannot be filled from the past) are dropped, as are duplicate
    dates (the last occurrence wins).
    """
    df = parse_date_column(df, date_col)
    df = df.sort_values("_date").drop_duplicates(subset="_date", keep="last")
    df = df.reset_index(drop=True)

    cols = [close_col, *sentiment_cols]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill: only past information is used.
    df[cols] = df[cols].ffill()
    df = df.dropna(subset=[close_col])
    if sentiment_cols:
        df = df.dropna(subset=list(sentiment_cols))
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"No rows remain after parsing dates and cleaning missing values "
            f"(date_col={date_col!r}, close_col={close_col!r})."
        )
    return df


def load_stock_csv(
    path: str | Path,
    *,
    date_col: str = "Date",
    close_col: str = "Close",
    sentiment_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load a historical price CSV with optional sentiment columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in sentiment_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sentiment column(s) not found in {path}: {missing}")
    return sort_and_clean(
        df, date_col=date_col, close_col=close_col, sentiment_cols=sentiment_cols
    )


def fetch_ticker_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download daily OHLCV history from Yahoo Finance (via yfinance)."""
    try:
        import yfinance as yf  # lazy import: only needed for --ticker
    except ImportError as exc:
        raise SystemExit(
            "--ticker requires yfinance. Install it with: pip install yfinance"
        ) from exc

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"yfinance returned no data for ticker {ticker!r}")
    if isinstance(data.columns, pd.MultiIndex):  # flatten (ticker, field) columns
        data.columns = data.columns.get_level_values(0)

    df = data.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(
        columns={
            "date": "Date",
            "close": "Close",
            "adj close": "Close",
            "adjclose": "Close",
        }
    )
    return sort_and_clean(df, date_col="Date", close_col="Close", sentiment_cols=())


def make_synthetic_data(
    n_days: int = 1500,
    seed: int = DEFAULT_SEED,
    include_sentiment: bool = True,
) -> pd.DataFrame:
    """Build a small synthetic series for end-to-end smoke tests.

    Prices follow a geometric random walk whose next-day return has a small
    predictable component driven by *yesterday's* sentiment signal, so the
    sentiment feature can genuinely help (and never leaks the future).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    sentiment = np.zeros(n_days)
    for t in range(1, n_days):
        sentiment[t] = 0.5 * sentiment[t - 1] + rng.normal(0.0, 0.5)

    returns = rng.normal(0.0005, 0.012, n_days)
    if include_sentiment:
        # Uses sentiment[t-1] only -> no look-ahead.
        returns[1:] += 0.004 * np.tanh(sentiment[:-1])

    close = 100.0 * np.exp(np.cumsum(returns))
    frame = {"Date": dates, "Close": close}
    if include_sentiment:
        frame["Sentiment"] = sentiment
    return pd.DataFrame(frame)


def resolve_sentiment_columns(
    df: pd.DataFrame,
    *,
    explicit: tuple[str, ...] = (),
    date_col: str = "Date",
    close_col: str = "Close",
) -> tuple[str, ...]:
    """Return sentiment feature columns.

    If ``explicit`` is given, those names are validated against the dataframe.
    Otherwise numeric-looking columns whose names match known sentiment tokens
    are auto-detected (e.g. ``sentiment``, ``polarity``, ``score``).
    """
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Sentiment column(s) not found in data: {missing}")
        return tuple(explicit)
    excluded = {date_col, close_col}
    return tuple(
        c for c in df.columns if SENTIMENT_COL_RE.search(str(c)) and c not in excluded
    )


def resolve_data_source(
    *,
    path: str | Path | None = None,
    ticker: str | None = None,
    synthetic: bool = False,
    seed: int = DEFAULT_SEED,
    date_col: str = "Date",
    close_col: str = "Close",
    sentiment_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, str]:
    """Load the dataset from a CSV, a ticker, or the synthetic generator.

    Returns ``(df, source_label)``.
    """
    chosen = sum(x is not None for x in (path, ticker)) + int(bool(synthetic))
    if chosen != 1:
        raise SystemExit(
            "Exactly one data source is required: pass --data FILE, "
            "--ticker SYMBOL, or --sample-synthetic."
        )
    if path is not None:
        df = load_stock_csv(
            path, date_col=date_col, close_col=close_col, sentiment_cols=sentiment_cols
        )
        return df, str(path)
    if ticker is not None:
        df = fetch_ticker_data(ticker)
        return df, f"yfinance:{ticker}"
    df = make_synthetic_data(seed=seed, include_sentiment=True)
    return df, "synthetic"


# ---------------------------------------------------------------------------
# Preprocessing: scaling + sliding windows
# ---------------------------------------------------------------------------

@dataclass
class WindowSet:
    """Sliding windows for one contiguous segment of the series."""

    X: np.ndarray  # (n_windows, lookback, n_features) float32 scaled
    y_scaled: np.ndarray  # (n_windows, horizon) float32 scaled close
    y_raw: np.ndarray  # (n_windows, horizon) raw price target
    prev_raw: np.ndarray  # (n_windows,) raw price at t-1 (for directional metrics)
    dates: np.ndarray | None  # (n_windows,) date of the *last* input row
    count: int


@dataclass
class PreparedData:
    """Everything needed to train/evaluate models for one dataset split."""

    feature_names: list[str]
    close_idx: int
    scaler: Any
    input_size: int
    train: WindowSet
    val: WindowSet
    test: WindowSet


def build_windows(
    feat_scaled: np.ndarray,
    close_raw: np.ndarray,
    dates: np.ndarray | None,
    *,
    lookback: int,
    horizon: int,
    close_idx: int = 0,
) -> WindowSet:
    """Build sliding input windows ``X`` and next-step targets ``y``.

    All values are derived from the *given segment only*; window ``i`` covers
    rows ``[i, i + lookback)`` and predicts the close at row ``i + lookback``.
    Vectorized (sliding-window view + fancy indexing) — semantics identical to
    the former per-window Python loop.
    """
    n = len(feat_scaled)
    if n <= lookback + horizon:
        raise ValueError(
            f"Not enough rows ({n}) for lookback={lookback}, horizon={horizon}."
        )
    count = n - lookback - horizon + 1
    X = np.ascontiguousarray(
        sliding_window_view(feat_scaled, lookback, axis=0)[:count].transpose(0, 2, 1)
    ).astype(np.float32, copy=False)

    target_idx = lookback + np.arange(count)[:, None] + np.arange(horizon)[None, :]
    y_scaled = np.asarray(feat_scaled[target_idx, close_idx], dtype=np.float32)
    y_raw = np.asarray(close_raw[target_idx], dtype=float)
    prev_raw = close_raw[lookback - 1 : lookback - 1 + count].copy()

    dates_out = None
    if dates is not None and len(dates) == n:
        dates_out = dates[lookback - 1 : lookback - 1 + count].copy()

    return WindowSet(
        X=X,
        y_scaled=y_scaled,
        y_raw=y_raw,
        prev_raw=prev_raw,
        dates=dates_out,
        count=count,
    )


def fit_scaler(rows: np.ndarray, kind: str = "minmax") -> Any:
    """Fit a scaler on ``rows`` (typically the training split only)."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if kind == "minmax":
        return MinMaxScaler().fit(rows)
    if kind == "standard":
        return StandardScaler().fit(rows)
    raise ValueError(f"Unknown scaler kind: {kind!r} (use 'minmax' or 'standard')")


def inverse_scale(values: np.ndarray, scaler: Any, col_idx: int) -> np.ndarray:
    """Map scaled feature values back to original units for one column."""
    values = np.asarray(values, dtype=float)
    if hasattr(scaler, "data_min_"):  # MinMaxScaler
        lo = scaler.data_min_[col_idx]
        hi = scaler.data_max_[col_idx]
        return values * (hi - lo) + lo
    if hasattr(scaler, "mean_"):  # StandardScaler
        return values * scaler.scale_[col_idx] + scaler.mean_[col_idx]
    raise TypeError(f"Unsupported scaler type: {type(scaler).__name__}")


def prepare_with_ranges(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    lookback: int,
    horizon: int,
    train_range: tuple[int, int],
    val_range: tuple[int, int],
    test_range: tuple[int, int],
    scaler_kind: str = "minmax",
    scaler: Any | None = None,
) -> PreparedData:
    """Prepare windows from explicit chronological row ranges.

    Unless a pre-fit ``scaler`` is injected, a scaler is fitted on the
    *training* rows only and then applied to the whole feature matrix; windows
    are built per segment so a window never spans split boundaries.
    """
    features = [data_cfg.close_col, *data_cfg.sentiment_cols]
    m = df[features].to_numpy(dtype=float)
    dates = df["_date"].to_numpy() if "_date" in df.columns else None

    a, b = train_range
    if scaler is None:
        scaler = fit_scaler(m[a:b], scaler_kind)
    scaled = scaler.transform(m)

    def window_set(rows: tuple[int, int]) -> WindowSet:
        a, b = rows
        seg_dates = dates[a:b] if dates is not None else None
        return build_windows(
            scaled[a:b],
            m[a:b, 0],
            seg_dates,
            lookback=lookback,
            horizon=horizon,
            close_idx=0,
        )

    return PreparedData(
        feature_names=features,
        close_idx=0,
        scaler=scaler,
        input_size=len(features),
        train=window_set(train_range),
        val=window_set(val_range),
        test=window_set(test_range),
    )


def prepare_dataframe(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = HORIZON,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    scaler_kind: str = "minmax",
) -> PreparedData:
    """Chronologically split ``df`` into train/val/test and build windows."""
    n = len(df)
    if n < 10:
        raise ValueError(f"Dataset too small ({n} rows).")
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train <= lookback + horizon:
        raise ValueError(
            f"Training split ({n_train} rows) is too small for lookback={lookback}."
        )
    return prepare_with_ranges(
        df,
        data_cfg=data_cfg,
        lookback=lookback,
        horizon=horizon,
        train_range=(0, n_train),
        val_range=(n_train, n_train + n_val),
        test_range=(n_train + n_val, n),
        scaler_kind=scaler_kind,
    )


# ---------------------------------------------------------------------------
# Metrics (original/unscaled price units)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev: np.ndarray | None = None,
) -> dict[str, float]:
    """Regression metrics on *original* price units.

    * MAE        - mean absolute error
    * RMSE       - root mean squared error
    * NRMSE      - RMSE normalized by the observed target range (ymax - ymin);
      falls back to NaN if the range is ~0
    * MAPE       - mean absolute percentage error (%)
    * R2         - coefficient of determination
    * directional_accuracy - fraction of steps where the sign of the change
      from ``prev`` matches between prediction and ground truth
      (``prev`` is the last observed close, i.e. ``y_{t-1}``).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("compute_metrics requires non-empty arrays.")

    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    y_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    nrmse = rmse / y_range if y_range > 1e-12 else float("nan")

    denom = np.abs(y_true)
    safe = np.where(denom > 1e-8, denom, np.nan)
    mape = float(np.nanmean(np.abs(err) / safe) * 100.0)

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    directional = float("nan")
    if prev is not None:
        prev = np.asarray(prev, dtype=float).ravel()
        if len(prev) == len(y_true):
            act_dir = np.sign(y_true - prev)
            pred_dir = np.sign(y_pred - prev)
            directional = float(np.mean(act_dir == pred_dir))

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional,
    }


# ---------------------------------------------------------------------------
# Models (PyTorch)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class BaselineGRU(nn.Module):
        """GRU forecaster: stacked GRU (``tanh``) + linear output head."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = DEFAULT_HIDDEN,
            num_layers: int = 1,
            dropout: float = DEFAULT_DROPOUT,
            horizon: int = HORIZON,
        ) -> None:
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.horizon = horizon
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, horizon)

        def forward(self, x: Any) -> Any:
            out, _ = self.gru(x)  # out: (B, T, H)
            return self.head(out[:, -1, :])

    class BaselineLSTM(nn.Module):
        """Stacked LSTM forecaster: LSTM + linear output head."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = DEFAULT_HIDDEN,
            num_layers: int = 2,
            dropout: float = DEFAULT_DROPOUT,
            horizon: int = HORIZON,
        ) -> None:
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.horizon = horizon
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, horizon)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

else:  # pragma: no cover - placeholder when torch is missing

    BaselineGRU = None  # type: ignore[assignment, misc]
    BaselineLSTM = None  # type: ignore[assignment, misc]


def build_model(name: str, input_size: int, cfg: ModelConfig) -> Any:
    """Construct a model by name (``gru`` | ``lstm``)."""
    require_torch()
    layers = cfg.num_layers or (2 if name == "lstm" else 1)
    if name == "gru":
        return BaselineGRU(
            input_size,
            hidden_size=cfg.hidden_size,
            num_layers=layers,
            dropout=cfg.dropout,
        )
    if name == "lstm":
        return BaselineLSTM(
            input_size,
            hidden_size=cfg.hidden_size,
            num_layers=layers,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model: {name!r} (use 'gru' or 'lstm')")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_network(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ModelConfig,
) -> dict[str, Any]:
    """Train with Adam + MSELoss and early stopping on validation loss.

    Returns a dict with the training history and the best-epoch state loaded
    back into ``model``.
    """
    require_torch()
    device = resolve_device(cfg.device)
    model = model.to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_ds)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vtotal += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = vtotal / len(val_ds)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience and epoch >= 2:
                break

    train_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_run": len(history),
        "train_time_s": train_time,
    }


def predict_model(
    model: Any,
    x: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """Numpy-in, numpy-out batched prediction."""
    require_torch()
    device = resolve_device(device)
    model = model.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i : i + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_model(model: Any, prep: PreparedData, cfg: ModelConfig) -> dict[str, Any]:
    """Predict on the test split, map back to original price units, score."""
    start = time.perf_counter()
    pred_scaled = predict_model(
        model, prep.test.X, batch_size=cfg.batch_size, device=cfg.device
    )
    infer_time = time.perf_counter() - start
    pred_raw = inverse_scale(pred_scaled, prep.scaler, prep.close_idx)
    metrics = compute_metrics(prep.test.y_raw, pred_raw, prep.test.prev_raw)
    return {"pred_raw": pred_raw, "metrics": metrics, "infer_time_s": infer_time}


def evaluate_persistence(prep: PreparedData) -> dict[str, Any]:
    """Naive persistence baseline: ``y_hat_t = y_{t-1}``."""
    start = time.perf_counter()
    pred_raw = prep.test.prev_raw.copy()
    infer_time = time.perf_counter() - start
    return {
        "pred_raw": pred_raw,
        "metrics": compute_metrics(prep.test.y_raw, pred_raw, prep.test.prev_raw),
        "infer_time_s": infer_time,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def save_metrics_json(out_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def append_metrics_csv(out_dir: Path, row: dict[str, Any]) -> Path:
    """Append one run's headline metrics as a CSV row (creates header if new)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.csv"
    frame = pd.DataFrame([{k: row.get(k) for k in _CSV_COLUMNS}])
    if not path.exists():
        frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, mode="a", header=False, index=False)
    return path


def save_plot(
    out_dir: Path,
    run_id: str,
    dates: np.ndarray | None,
    actual: np.ndarray,
    pred: np.ndarray,
    title: str = "Actual vs predicted (test split)",
) -> Path | None:
    """Save a simple actual-vs-predicted PNG (skips gracefully if no matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] matplotlib unavailable; skipping plot ({exc})")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if dates is not None:
        ax.plot(dates, actual, label="actual", lw=1.2)
        ax.plot(dates, pred, label="predicted", lw=1.2)
    else:
        ax.plot(actual, label="actual")
        ax.plot(pred, label="predicted")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = out_dir / f"{run_id}_plot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(payload: dict[str, Any]) -> None:
    """Human-readable console summary."""
    res = payload["results"]
    print("\n" + "=" * 72)
    print("BASELINE RUN SUMMARY")
    print("=" * 72)
    print(f"run_id       : {payload['run_id']}")
    print(f"data_source  : {payload['data_source']}  ({payload['n_rows']} rows)")
    cfg = payload["config"]["model"]
    print(f"model        : {cfg['model']} (hidden={cfg['hidden_size']}, "
          f"layers={cfg['num_layers'] or 'auto'}, seed={cfg['seed']})")
    print(f"lookback     : {payload['config']['run']['lookback']}")

    if "persistence" in res:
        m = res["persistence"]["metrics"]
        print("\n-- Persistence (y_hat_t = y_{t-1}) --")
        print(f"   MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.3f}%  "
              f"R2={m['r2']:.4f}  DirAcc={m['directional_accuracy']:.4f}")

    if "model" in res:
        m = res["model"]["metrics"]
        print("\n-- Model --")
        print(f"   best_epoch={res['model']['best_epoch']}  "
              f"val_loss={res['model']['best_val_loss']:.6f}")
        print(f"   MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MAPE={m['mape']:.3f}%  "
              f"R2={m['r2']:.4f}  DirAcc={m['directional_accuracy']:.4f}")
        print(f"   train_time={res['model']['train_time_s']:.2f}s  "
              f"infer_time={res['model']['infer_time_s']:.4f}s")

    save_ = payload.get("saved")
    if save_:
        print("\nsaved:")
        for key, val in save_.items():
            if val:
                print(f"   {key}: {val}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_baseline(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    run_cfg: RunConfig,
    model_cfg: ModelConfig,
    source_label: str,
) -> dict[str, Any]:
    """Execute one full baseline run and persist its results."""
    set_seed(model_cfg.seed)
    prep = prepare_dataframe(
        df,
        data_cfg=data_cfg,
        lookback=run_cfg.lookback,
        horizon=run_cfg.horizon,
        val_frac=run_cfg.val_frac,
        test_frac=run_cfg.test_frac,
        scaler_kind=run_cfg.scaler,
    )

    run_id = make_run_id("baseline")
    results: dict[str, Any] = {}

    if run_cfg.run_persistence:
        per = evaluate_persistence(prep)
        results["persistence"] = {
            "metrics": per["metrics"],
            "infer_time_s": per["infer_time_s"],
        }

    model = build_model(model_cfg.model, prep.input_size, model_cfg)
    history = train_network(
        model, prep.train.X, prep.train.y_scaled, prep.val.X, prep.val.y_scaled,
        model_cfg,
    )
    ev = evaluate_model(model, prep, model_cfg)

    results["model"] = {
        "name": model_cfg.model,
        "metrics": ev["metrics"],
        "train_time_s": history["train_time_s"],
        "infer_time_s": ev["infer_time_s"],
        "best_epoch": history["best_epoch"],
        "epochs_run": history["epochs_run"],
        "best_val_loss": history["best_val_loss"],
    }

    payload: dict[str, Any] = {
        "run_id": run_id,
        "project": "improving-gru-sentiment-stock-forecasting-padmakumari-maam",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": source_label,
        "n_rows": len(df),
        "config": {
            "model": asdict(model_cfg),
            "run": asdict(run_cfg),
            "data": asdict(data_cfg),
        },
        "results": results,
    }

    saved = {
        "json": save_metrics_json(run_cfg.out_dir, run_id, payload),
        "csv": None,
        "plot": None,
    }

    if run_cfg.run_persistence:
        row: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": payload["timestamp"],
            "model": model_cfg.model,
            "data_source": source_label,
            "n_rows": len(df),
            "lookback": run_cfg.lookback,
            "hidden_size": model_cfg.hidden_size,
            "num_layers": model_cfg.num_layers or (2 if model_cfg.model == "lstm" else 1),
            "epochs_run": history["epochs_run"],
            "best_epoch": history["best_epoch"],
            "train_time_s": history["train_time_s"],
            "infer_time_s": ev["infer_time_s"],
        }
        row.update({f"persistence_{k}": v for k, v in results["persistence"]["metrics"].items()})
        row.update(results["model"]["metrics"])
        saved["csv"] = append_metrics_csv(run_cfg.out_dir, row)

    if run_cfg.plot:
        saved["plot"] = save_plot(
            run_cfg.out_dir,
            run_id,
            prep.test.dates,
            prep.test.y_raw[:, 0],
            ev["pred_raw"][:, 0],
            title=f"Actual vs predicted (test split) - {model_cfg.model}",
        )

    payload["saved"] = {k: str(v) if v else None for k, v in saved.items()}
    # Re-write JSON now that saved paths are known.
    save_metrics_json(run_cfg.out_dir, run_id, payload)

    print_summary(payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.baseline",
        description=(
            "Baseline stock forecasting: persistence, GRU, and LSTM with "
            "strict time-series hygiene and leakage-free evaluation."
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.baseline --sample-synthetic\n"
            "  python -m src.baseline --data data/prices.csv --sentiment-cols Sentiment\n"
            "  python -m src.baseline --ticker AAPL --model lstm --plot\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = parser.add_argument_group("data source (exactly one required)")
    data.add_argument("--data", metavar="FILE", help="path to a price CSV")
    data.add_argument("--ticker", metavar="SYMBOL", help="Yahoo Finance ticker (yfinance)")
    data.add_argument("--sample-synthetic", action="store_true",
                      help="generate a small synthetic series for testing")
    data.add_argument("--date-col", default="Date", help="date column name (default: Date)")
    data.add_argument("--close-col", default="Close", help="close column name (default: Close)")
    data.add_argument("--sentiment-cols", default="", metavar="COLS",
                      help="comma-separated sentiment feature columns (default: auto-detect)")
    data.add_argument("--no-sentiment", action="store_true",
                      help="disable sentiment features entirely (price-only)")

    win = parser.add_argument_group("windowing / splitting")
    win.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="window length (default: 30)")
    win.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    win.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    win.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")

    mod = parser.add_argument_group("model / training")
    mod.add_argument("--model", choices=["gru", "lstm"], default="gru")
    mod.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    mod.add_argument("--layers", type=int, default=None, help="default: 1 (GRU) / 2 (LSTM)")
    mod.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    mod.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    mod.add_argument("--lr", type=float, default=DEFAULT_LR)
    mod.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    mod.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    mod.add_argument("--seed", type=int, default=DEFAULT_SEED)
    mod.add_argument("--device", default="auto", help="auto | cpu | cuda")

    out = parser.add_argument_group("output")
    out.add_argument("--out-dir", type=Path, default=Path("results"))
    out.add_argument("--plot", action="store_true", help="save an actual-vs-predicted PNG")
    out.add_argument("--no-persistence", action="store_true", help="skip the persistence baseline")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.no_sentiment:
        sentiment_cols: tuple[str, ...] = ()
    else:
        explicit = tuple(c.strip() for c in args.sentiment_cols.split(",") if c.strip())
        sentiment_cols = explicit  # validated against the loaded data below

    df, source_label = resolve_data_source(
        path=args.data,
        ticker=args.ticker,
        synthetic=args.sample_synthetic,
        seed=args.seed,
        date_col=args.date_col,
        close_col=args.close_col,
        sentiment_cols=sentiment_cols,
    )

    if not args.no_sentiment and not explicit:
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
    run_cfg = RunConfig(
        lookback=args.lookback,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        scaler=args.scaler,
        run_persistence=not args.no_persistence,
        plot=args.plot,
        out_dir=args.out_dir,
    )

    run_baseline(df, data_cfg=data_cfg, run_cfg=run_cfg,
                 model_cfg=model_cfg, source_label=source_label)


if __name__ == "__main__":
    main()

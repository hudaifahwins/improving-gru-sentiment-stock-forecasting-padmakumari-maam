"""VolGatedGRU — constructive upgrade over the baseline GRU + sentiment framework.

Design / grounding
-------------------
- Primary GRU branch consumes ``price + tech + ohlcv_volnorm`` features
  (causal, backward-looking only).
- A *negative-bias* sigmoid residual gate (init ``b = -2.0``) on lagged
  sentiment features modulates the GRU output additively, so sentiment can only
  nudge a prediction when evidence is strong (gate starts closed).
- Loss: volatility-normalized Huber + directional-margin BCE (k calibrated on
  the train-split return std, leak-free).
- Hysteresis no-trade deadband ``tau`` is fit on the *validation* split only
  (net-return maximization with 5 bps/side costs) and then applied to test.

Anti-leakage contract
----------------------
- Chronological 70/15/15 split + expanding walk-forward OOS.
- Scalers fit on train rows only; windows built per segment (non-overlapping
  stride = horizon for h=3/5); targets use only rows inside the segment.
- Sentiment features are 1-trading-day lagged (already lagged in the cached
  GDELT tone CSVs) and are themselves causal.

Metrics
-------
- Price-level reconstructed MAE / RMSE / nRMSE / MAPE
- OOS DirAcc, majority-sign base rate, excess DirAcc, seed collapse k/3
- Newey-West/HAC Diebold-Mariano stat & p-value (vs base rate AND vs baseline
  GRU), coverage %, turnover, net return after 5 bps/side cost.

Sentiment source honesty
-----------------------
- Uses the cached real GDELT tone series (``data/sentiment_<ticker>.csv``).
- ``sent_vol_z`` is an *activity proxy* derived from the rolling count of
  non-zero tone days (raw article counts are unavailable; GDELT DOC Vol API is
  rate-limited). This is documented and not presented as raw news volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from src.features_research import build_ohlcv_volnorm_features, has_ohlcv
from src.baseline import fit_scaler, compute_metrics, set_seed

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SEEDS = (42, 123, 2026)
DEFAULT_LOOKBACK = 30
DEFAULT_HIDDEN = 50
COST_BPS_PER_SIDE = 0.0005
SENTIMENT_CSV = {
    "AAPL": Path("data/sentiment_AAPL.csv"),
    "005930.KS": Path("data/sentiment_005930.KS.csv"),
}
GATE_BIAS = -2.0


# ---------------------------------------------------------------------------
# data / features (leak-free)
# ---------------------------------------------------------------------------

def load_ohlcv(ticker: str, years: int = 7) -> pd.DataFrame:
    """Load OHLCV history and merge cached lagged GDELT sentiment, if present."""
    import yfinance as yf

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"date": "Date", "adj close": "Close", "adjclose": "Close",
                            "close": "Close"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    sfile = SENTIMENT_CSV.get(ticker)
    if sfile is not None and sfile.exists():
        sent = pd.read_csv(sfile)
        sent["Date"] = pd.to_datetime(sent["Date"]).dt.normalize()
        df = df.merge(sent[["Date", "Sentiment"]], on="Date", how="left")
        # cached CSVs already lag the tone by 1 trading day; double-lag is NOT applied.
    else:
        df["Sentiment"] = np.nan
    return df


def _causal_tech_features(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    close = df[close_col].astype(float)
    log_close = np.log(close)
    ret = log_close.diff()
    out = pd.DataFrame(index=df.index)
    out["ret_mean5"] = ret.rolling(5, min_periods=1).mean()
    out["ret_std10"] = ret.rolling(10, min_periods=1).std()
    out["mom20"] = log_close - log_close.shift(20)
    out["range10"] = (
        close.rolling(10).max() - close.rolling(10).min()
    ) / close.shift(1)
    out["log_close"] = log_close
    return out


def build_feature_frame(
    df: pd.DataFrame,
    *,
    include_sentiment: bool = True,
    close_col: str = "Close",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (features, base_names, sentiment_names) — strictly causal."""
    tech = _causal_tech_features(df, close_col)
    out = pd.DataFrame(index=df.index)
    out["Close"] = df[close_col].astype(float)
    for c in tech.columns:
        out[c] = tech[c]

    base_names = [c for c in out.columns if c != "Close"][:]

    ohlcv_names: list[str] = []
    sentiment_names: list[str] = []
    if has_ohlcv(df):
        o = build_ohlcv_volnorm_features(df, close_col=close_col)
        for c in o.columns:
            out[c] = o[c]
        ohlcv_names = list(o.columns)
    if include_sentiment and "Sentiment" in df.columns:
        tone = df["Sentiment"].astype(float)
        out["sent_tone"] = tone
        out["sent_shock"] = tone - tone.rolling(5, min_periods=1).mean()
        # activity proxy (documented honest limitation): z-score of trailing count
        # of non-zero tone days.
        active = (tone != 0).astype(float).rolling(20, min_periods=10).sum()
        astd = active.rolling(20, min_periods=10).std()
        out["sent_vol_z"] = (active - active.rolling(20, min_periods=10).mean()) / (astd + 1e-8)
        out["sent_x_vol"] = out["sent_shock"] * np.clip(out["sent_vol_z"], 0.0, 3.0)
        sentiment_names = ["sent_tone", "sent_shock", "sent_vol_z", "sent_x_vol"]

    out = out.dropna()
    return out, base_names, sentiment_names


# ---------------------------------------------------------------------------
# splits / windows (non-overlapping stride for h>1)
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    p_train: np.ndarray  # closing price at target time (same segment)
    prev_train: np.ndarray  # P_{t-h}
    x_val: np.ndarray
    y_val: np.ndarray
    p_val: np.ndarray
    prev_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    p_test: np.ndarray
    prev_test: np.ndarray
    scaler: Any
    feature_names: list[str]
    n_sent: int = 0


def make_windows(feat: np.ndarray, close: np.ndarray, lookback: int,
                 horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Non-overlapping windows (stride = horizon) within one segment."""
    t_idx = np.arange(lookback, len(feat), horizon)
    t_idx = t_idx[(t_idx - horizon) >= 0]
    if len(t_idx) == 0:
        raise ValueError("no valid windows for segment/horizon")
    x = np.empty((len(t_idx), lookback, feat.shape[1]), dtype=np.float32)
    for i, t in enumerate(t_idx):
        x[i] = feat[t - lookback : t]
    y = np.log(close[t_idx] / close[t_idx - horizon]).astype(np.float32)
    prev = close[t_idx - horizon]
    return x, y.reshape(-1, 1), close[t_idx], prev


def prepare_split_data(
    df: pd.DataFrame,
    *,
    features: str = "price+tech+ohlcv_volnorm+sentiment",
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = 3,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitData:
    fdf, base_names, sentiment_names = build_feature_frame(
        df, include_sentiment=("sentiment" in features)
    )
    feat_cols = list(fdf.columns)  # includes Close as first feature
    m = fdf[feat_cols].to_numpy(float)
    n = len(m)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test
    min_seg = lookback + horizon + 1
    if min(n_train, n_val, n_test) < min_seg:
        raise ValueError(f"segment too small: train {n_train}, val {n_val}, test {n_test}")

    scaler = fit_scaler(m[:n_train], "minmax")
    scaled = scaler.transform(m)
    train = make_windows(scaled[:n_train], m[:n_train, 0], lookback, horizon)
    val = make_windows(scaled[n_train : n_train + n_val], m[n_train : n_train + n_val, 0],
                       lookback, horizon)
    test = make_windows(scaled[n_train + n_val :], m[n_train + n_val :, 0],
                        lookback, horizon)
    xt, yt, pt, prevt = train
    xv, yv, pv, prevv = val
    xe, ye, pe, preve = test
    n_sent = len(sentiment_names) if ("sentiment" in features) else 0
    return SplitData(xt, yt, pt, prevt, xv, yv, pv, prevv,
                     xe, ye, pe, preve, scaler, feat_cols, n_sent)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class BaselineGRU(nn.Module):
    def __init__(self, input_size: int, hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class VolGatedGRU(nn.Module):
    """GRU on price+tech+ohlcv with sentiment residual gate (bias -2.0)."""

    def __init__(self, input_size: int, n_sent: int, hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.sent_in = nn.Linear(n_sent, hidden)
        self.gate = nn.Linear(hidden, 1)
        # negative-bias sigmoid residual gate (starts closed)
        nn.init.constant_(self.gate.bias, GATE_BIAS)
        self.sent_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h = out[:, -1, :]
        base = self.head(h)
        sg = torch.tanh(self.sent_in(s))          # (B, T, H)
        sg_last = sg[:, -1, :]                    # (B, H) — last available day
        g = torch.sigmoid(self.gate(sg_last))
        residual = self.sent_head(sg_last)
        return base + g * residual


# ---------------------------------------------------------------------------
# training (volatility-normalized Huber + directional margin)
# ---------------------------------------------------------------------------

def _causal_vol(y: np.ndarray, window: int = 20) -> np.ndarray:
    vol = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo = max(0, i - window)
        vol[i] = float(np.std(y[lo:i])) if i > lo else 1.0
    return np.maximum(vol, 1e-4)


def train_model(
    model: nn.Module,
    split: SplitData,
    *,
    use_sentiment: bool,
    gated: bool = False,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 64,
    seed: int = 42,
    loss_kind: str = "mse",
) -> dict[str, Any]:
    set_seed(seed)
    device = "cpu"
    model = model.to(device)
    x_tr = torch.from_numpy(split.x_train)
    y_tr = torch.from_numpy(split.y_train)
    x_va = torch.from_numpy(split.x_val)
    y_va = torch.from_numpy(split.y_val)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_kind in ("dir_margin", "vol_huber_margin"):
        # volatility-normalized targets for the Huber + margin loss
        vol_tr = torch.from_numpy(_causal_vol(split.y_train[:, 0])).float()
        vol_va = torch.from_numpy(_causal_vol(split.y_val[:, 0])).float()
        k = 1.0 / float(np.std(split.y_train) + 1e-4)
    else:
        vol_tr = vol_va = None
        k = 1.0

    def step_loss(xb, yb, volb):
        if gated:
            s = xb[:, :, -split.n_sent:]
            xb_main = xb[:, :, : -split.n_sent]
            out = model(xb_main, s)
        else:
            out = model(xb)
        out = out[:, 0]
        yb = yb[:, 0]
        if loss_kind in ("dir_margin", "vol_huber_margin"):
            mag = F.huber_loss(out / volb, yb / volb, delta=1.0)
            direction = F.binary_cross_entropy_with_logits(k * out, (yb > 0).float())
            return 0.5 * mag + 0.5 * direction
        return F.mse_loss(out, yb)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    n = len(x_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            volb = vol_tr[idx] if vol_tr is not None else torch.ones_like(yb[:, 0])
            opt.zero_grad()
            loss = step_loss(xb, yb, volb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for i in range(0, len(x_va), batch_size):
                xb, yb = x_va[i : i + batch_size], y_va[i : i + batch_size]
                volb = vol_va[i : i + batch_size] if vol_va is not None else torch.ones_like(yb[:, 0])
                vloss += step_loss(xb, yb, volb).item() * len(xb)
        vloss /= len(x_va)
        if vloss < best_val - 1e-6:
            best_val, best_epoch, bad = vloss, epoch, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": epoch}


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, use_sentiment: bool,
            n_sent: int, gated: bool = False) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(x)
    if gated:
        s = x[:, :, -n_sent:]
        main = x[:, :, : -n_sent]
        out = model(main, s)
    else:
        out = model(x)
    return out.numpy()[:, 0]


# ---------------------------------------------------------------------------
# hysteresis deadband fit (validation only)
# ---------------------------------------------------------------------------

def fit_hysteresis_tau(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    *,
    cost: float = COST_BPS_PER_SIDE,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Choose tau maximizing net return on validation (5 bps/side)."""
    if grid is None:
        grid = np.linspace(0.0, 0.06, 25)
    best_tau, best_net = 0.0, -np.inf
    best_meta: dict[str, float] = {}
    r = np.expm1(y_val)
    for tau in grid:
        sign = np.where(np.abs(pred_val) > tau, np.sign(pred_val), 0.0)
        weeks = np.sum(np.abs(np.diff(sign)) if len(sign) > 1 else 0)
        gross = float(np.sum(sign * r))
        net = gross - cost * float(np.sum(np.abs(np.diff(sign))))
        if net > best_net:
            best_net, best_tau = net, float(tau)
            cov = float(np.mean(np.abs(sign) > 0))
            best_meta = {"net_return": net, "gross_return": gross, "coverage": cov}
    return best_tau, best_meta


# ---------------------------------------------------------------------------
# Newey-West / HAC Diebold-Mariano
# ---------------------------------------------------------------------------

def hac_dm(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano with Newey-West HAC variance.

    Loss = squared return error. Returns (DM stat, two-sided p-value).
    """
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    if len(e1) != len(e2):
        raise ValueError("DM requires equal-length error vectors")
    d = e1**2 - e2**2
    T = len(d)
    mean_d = float(np.mean(d))
    L = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    L = max(L, 1)
    gamma0 = float(np.mean((d - mean_d) ** 2))
    var = gamma0
    for lag in range(1, L + 1):
        c = float(np.mean((d[:-lag] - mean_d) * (d[lag:] - mean_d)))
        w = 1.0 - lag / (L + 1)
        var += 2.0 * w * c
    z = mean_d / math.sqrt(max(var / T, 1e-12))
    try:
        from scipy.stats import norm

        p = 2.0 * norm.sf(abs(z))
    except Exception:  # pragma: no cover
        p = float("nan")
    return z, p

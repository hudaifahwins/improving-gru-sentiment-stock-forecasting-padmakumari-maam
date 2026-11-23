"""Causal OHLCV microstructure features (research extension).

Reference grounding:
- Garman & Klass (1980) — range-based volatility estimators.
- Blume, Easley & O'Hara (1994) — volume-return dynamics.

All features are strictly backward-looking: any value at row ``j`` depends only
on rows ``<= j``. Window construction in the horizon experiments slices
``feat[t-lookback:t]``, so a model at row ``t`` only sees rows ``< t``.

If an OHLCV column is missing (or all-zero volume), the corresponding feature
is omitted from the returned frame and the caller can report the gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Candidate column names (case-insensitive) in order of preference.
_CANDIDATES = {
    "open": ["Open", "open", "OPEN"],
    "high": ["High", "high", "HIGH"],
    "low": ["Low", "low", "LOW"],
    "close": ["Close", "close", "CLOSE"],
    "volume": ["Volume", "volume", "VOLUME"],
}


def _find_col(df: pd.DataFrame, key: str) -> str | None:
    for cand in _CANDIDATES[key]:
        if cand in df.columns:
            return cand
    return None


def has_ohlcv(df: pd.DataFrame) -> bool:
    return all(_find_col(df, k) is not None for k in ("open", "high", "low", "close"))


def build_ohlcv_volnorm_features(
    df: pd.DataFrame, *, close_col: str = "Close"
) -> pd.DataFrame:
    """Return strictly causal OHLCV features indexed like ``df``.

    Output columns:
      ret_volnorm  r_t / (std_20(r) + 1e-8)
      hl_range     (High_t - Low_t) / (Close_t + 1e-8)
      clv         (2*Close_t - High_t - Low_t) / (High_t - Low_t + 1e-8)
      vol_z20     (log1p(Vol_t) - mean_20) / (std_20 + 1e-8)
      mom_5_volnorm  5-day cumulative log return / (sqrt(5) * std_20(r) + 1e-8)
    """
    close_src = _find_col(df, "close") or close_col
    close = df[close_src].astype(float)
    out = pd.DataFrame(index=df.index)

    log_close = np.log(close)
    ret = log_close.diff()
    std20 = ret.rolling(20, min_periods=10).std()
    out["ret_volnorm"] = ret / (std20 + 1e-8)

    high = df[_find_col(df, "high")].astype(float) if _find_col(df, "high") else np.nan
    low = df[_find_col(df, "low")].astype(float) if _find_col(df, "low") else np.nan
    if isinstance(high, pd.Series) and isinstance(low, pd.Series):
        out["hl_range"] = (high - low) / (close + 1e-8)
        rng = high - low
        out["clv"] = (2 * close - high - low) / (rng + 1e-8)
    else:
        out["hl_range"] = 0.0
        out["clv"] = 0.0

    vol_col = _find_col(df, "volume")
    if vol_col is not None:
        vol = df[vol_col].astype(float)
        lv = np.log1p(vol)
        out["vol_z20"] = (lv - lv.rolling(20, min_periods=10).mean()) / (
            lv.rolling(20, min_periods=10).std() + 1e-8
        )
    else:
        out["vol_z20"] = 0.0

    out["mom_5_volnorm"] = (log_close - log_close.shift(5)) / (
        np.sqrt(5) * std20 + 1e-8
    )
    return out

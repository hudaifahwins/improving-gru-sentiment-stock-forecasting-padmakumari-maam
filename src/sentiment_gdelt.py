"""Fetch real historical news-tone sentiment from the GDELT public API.

Uses GDELT DOC 2.1 ``TimelineTone`` (no API key). Outputs a trading-day-aligned
CSV with a *lagged* sentiment column (tone covering the previous day is used at
day t, so no look-ahead occurs relative to the close of day t).

Usage::
    python -m src.sentiment_gdelt --ticker AAPL --query AAPL --out data/sentiment_AAPL.csv
    python -m src.sentiment_gdelt --ticker 005930.KS --query "Samsung Electronics" \\
        --out data/sentiment_005930.KS.csv

If the API is unreachable/rate-limited, the script exits 1 with the exact
failure, and no file is written.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_tone_timeline(query: str, start: date, end: date, timeout: int = 60):
    """Return date->tone Series from GDELT TimelineTone."""
    if requests is None:
        raise RuntimeError("requests not installed: pip install requests")
    params = {
        "query": query,
        "mode": "TimelineTone",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "output": "json",
    }
    resp = requests.get(BASE, params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GDELT HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    timeline = data.get("timeline")
    if isinstance(timeline, dict):
        if "date" not in timeline:
            raise RuntimeError("GDELT response missing 'timeline.date' key: " + str(data)[:300])
        dates = pd.to_datetime(timeline["date"], format="%Y%m%d")
        tone = np.asarray(pd.to_numeric(timeline["value"], errors="coerce"))
        return pd.Series(tone, index=dates).sort_index()
    if isinstance(timeline, list):
        series = next((s for s in timeline if s.get("series") == "Average Tone"), None)
        if series is None or "data" not in series:
            raise RuntimeError("GDELT timeline missing 'Average Tone' series: " + str(data)[:300])
        dates = pd.to_datetime([d["date"] for d in series["data"]], format="%Y%m%dT%H%M%SZ")
        tone = np.asarray(pd.to_numeric([d["value"] for d in series["data"]], errors="coerce"))
        return pd.Series(tone, index=dates).sort_index()
    raise RuntimeError("GDELT response has unexpected timeline type: " + str(data)[:300])


def make_sentiment_csv(ticker: str, query: str, out: Path, days: int = 3650) -> Path:
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=days)
    tone = fetch_tone_timeline(query, start, end)

    prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if prices is None or len(prices) == 0:
        raise RuntimeError(f"yfinance returned no price data for {ticker!r}")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    closes = prices["Close"].dropna()

    frame = pd.DataFrame({"Date": closes.index, "Close": closes.values})
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    tone.index = pd.to_datetime(tone.index).normalize()
    # Coverage stats (honest reporting).
    overlap = len(set(frame["Date"]) & set(tone.index))
    # Lag tone by one calendar-before-trading-day step: tone known before close of t.
    frame = frame.merge(tone.rename("Sentiment_raw"), left_on="Date", right_index=True,
                        how="left")
    frame["Sentiment"] = frame["Sentiment_raw"].shift(1)
    frame = frame.drop(columns=["Sentiment_raw"]).dropna(subset=["Sentiment"])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}: rows={len(frame)} price_days={len(closes)} "
          f"tone_days={len(tone)} aligned_lag_rows={len(frame)}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m src.sentiment_gdelt")
    p.add_argument("--ticker", required=True)
    p.add_argument("--query", required=True, help="GDELT query keywords (English)")
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=int, default=3650)
    args = p.parse_args(argv)
    try:
        make_sentiment_csv(args.ticker, args.query, args.out, args.days)
    except Exception as exc:  # report exact failure; never fake data
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

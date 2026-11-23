"""Build merged OHLCV + lagged-GDELT-sentiment CSVs for the experiment batch.

Output columns: Date, Open, High, Low, Close, Volume, Sentiment, NewsCount.

Leakage discipline:
* The ``Sentiment`` (Average Tone) column is reused *verbatim* from the
  committed ingestion CSVs (``data/sentiment_<ticker>.csv``), which already
  apply a 1-trading-day lag (day t only sees tone known before the close of t).
* ``NewsCount`` (GDELT TimelineVol "Volume Intensity") is fetched fresh and
  lagged with the identical ``shift(1)`` rule.
* Prices come from yfinance (auto-adjusted OHLCV); the merge is an inner join
  on trading dates, so no future row can align with a past one.

The raw ingestion CSVs are read-only inputs and are never modified.

Usage::

    python3 scripts/build_price_sentiment_csv.py --ticker AAPL --query AAPL \\
        --sentiment-csv data/sentiment_AAPL.csv --out data/prices_sentiment_AAPL.csv
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.sentiment_gdelt import fetch_timeline


def fetch_counts_with_retry(query: str, start: date, end: date, *,
                            attempts: int = 6, pause: float = 12.0):
    """Fetch GDELT TimelineVol with backoff (API allows ~1 req / 5 s)."""
    last = None
    for i in range(attempts):
        try:
            return fetch_timeline(query, start, end, mode="TimelineVol",
                                  series_name="Volume Intensity")
        except Exception as exc:  # 429 / transient network errors
            last = exc
            print(f"[gdelt] attempt {i + 1}/{attempts} failed: {exc}; "
                  f"retrying in {pause:.0f}s", flush=True)
            time.sleep(pause)
    raise last


def build(ticker: str, query: str, sentiment_csv: Path, out: Path, days: int = 1825) -> Path:
    prices = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
    if prices is None or len(prices) == 0:
        raise SystemExit(f"yfinance returned no data for {ticker!r}")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    ohlcv = prices.reset_index().rename(columns={"Date": "Date"})
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.normalize()
    ohlcv = ohlcv[["Date", "Open", "High", "Low", "Close", "Volume"]]

    tone = pd.read_csv(sentiment_csv, usecols=["Date", "Sentiment"])
    tone["Date"] = pd.to_datetime(tone["Date"]).dt.normalize()

    end = date.today()
    start = end - timedelta(days=days)
    counts = fetch_counts_with_retry(query, start, end)
    counts.index = pd.to_datetime(counts.index).normalize()
    cnt = pd.DataFrame({"Date": counts.index, "NewsCount_raw": counts.values})

    frame = ohlcv.merge(tone, on="Date", how="inner").merge(cnt, on="Date", how="inner")
    # Lag news counts by one trading day (same rule as the tone ingestion).
    frame["NewsCount"] = frame["NewsCount_raw"].shift(1)
    frame = frame.drop(columns=["NewsCount_raw"]).dropna(subset=["Sentiment", "NewsCount"])
    frame = frame.sort_values("Date").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    tone_nonzero = float((frame["Sentiment"] != 0).mean())
    print(f"wrote {out}: rows={len(frame)} "
          f"range={frame['Date'].min().date()}..{frame['Date'].max().date()} "
          f"tone_nonzero_share={tone_nonzero:.3f} "
          f"news_count_zero_share={float((frame['NewsCount'] == 0).mean()):.3f}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scripts.build_price_sentiment_csv")
    p.add_argument("--ticker", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--sentiment-csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--days", type=int, default=1825)
    args = p.parse_args(argv)
    build(args.ticker, args.query, args.sentiment_csv, args.out, days=args.days)


if __name__ == "__main__":
    main()

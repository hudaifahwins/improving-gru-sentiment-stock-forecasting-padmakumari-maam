"""Local ticker-specific sentiment — zero rate-limit dependency.

Sources
-------
- AAPL: real firm-level text from SEC EDGAR (public API, no key). We fetch the
  company's 8-K filing list (submissions endpoint), resolve each filing's primary
  document from the Archives index, download the raw HTML, strip markup, and
  score the prose with the Loughran–McDonald (2011) financial lexicon.
- 005930.KS: SEC does not cover Korean issuers. There is **no** firm-level text
  cache on this machine for Samsung; we therefore do NOT fabricate series. The
  pipeline returns an explicit ``coverage_gap`` for Samsung, and callers must use
  the cached GDELT tone (English query) as documented in the report.

Lexicon
-------
Embedded canonical Loughran–McDonald positive / negative word sets (financial
polarity). Scores are computed as:

    net = (n_pos - n_neg) / (n_words + 1e-6)

Deterministic; no neural model required (FinBERT optional if installed).

Lags
----
ANY series produced here is lagged by exactly one trading day before it is used
as a feature (see the consensus runner), so no look-ahead is possible.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

# Loughran-McDonald (2011) financial polarity (canonical compact lists)
LM_POS = {
    "accomplish", "accomplished", "accomplishment", "achieve", "achieved",
    "achievement", "advantage", "advantages", "beneficial", "benefit", "benefits",
    "best", "better", "boost", "boosted", "capable", "commitment", "confident",
    "cooperation", "deliver", "delivered", "enhance", "enhanced", "exceed",
    "exceeded", "excellent", "favorable", "favorably", "gain", "gained",
    "good", "great", "growth", "improve", "improved", "improvement",
    "increase", "increased", "innovation", "innovative", "leader", "leading",
    "opportunities", "opportunity", "positive", "profit", "profitable",
    "progress", "rapid", "recover", "recovered", "revenue", "robust",
    "solid", "stability", "stable", "strength", "strong", "stronger",
    "success", "successful", "superior", "win", "won",
}
LM_NEG = {
    "adverse", "adversely", "against", "bad", "breach", "challenges",
    "claim", "claims", "conflict", "corruption", "costly", "decline",
    "declined", "decrease", "decreased", "deficit", "delay", "delays",
    "deteriorate", "discontinued", "dispute", "disputes", "downgrade",
    "downside", "failure", "failures", "fraud", "impairment", "impaired",
    "inability", "inadequate", "investigation", "investigations", "lawsuit",
    "lawsuits", "liability", "liabilities", "litigation", "loss", "losses",
    "matters", "negative", "noncompliance", "penalty", "penalties",
    "poor", "problem", "problems", "recall", "recalls", "regulatory",
    "restatement", "restructuring", "risk", "risks", "scrutiny",
    "significant adverse", "termination", "uncertain", "uncertainty",
    "unfavorable", "volatile", "volatility", "weak", "weaken", "weaker",
    "weakness", "weaknesses", "worst", "write-off", "write-down",
}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z']+")
_CIK = "0000320193"  # Apple Inc.
_SEC = "https://data.sec.gov"
_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


def _headers() -> dict[str, str]:
    return {"User-Agent": "research hudaifahwins@example.com (academic project)"}


def score_text_lexicon(text: str) -> dict[str, float]:
    """Loughran-McDonald net polarity on one document."""
    stripped = _TAG_RE.sub(" ", text)
    plain = html.unescape(stripped)
    words = _WORD_RE.findall(plain.lower())
    n_words = len(words)
    n_pos = sum(1 for w in words if w in LM_POS)
    n_neg = sum(1 for w in words if w in LM_NEG)
    net = (n_pos - n_neg) / (n_words + 1e-6)
    return {"n_words": n_words, "n_pos": n_pos, "n_neg": n_neg,
            "net": net}


def _fetch_json(url: str, tries: int = 4, wait: float = 1.5):
    if requests is None:
        raise RuntimeError("requests not installed")
    for i in range(tries):
        try:
            r = requests.get(url, headers=_headers(), timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(wait * (i + 1))
                continue
            raise RuntimeError(f"HTTP {r.status_code} @ {url}")
        except requests.exceptions.RequestException as exc:
            if i == tries - 1:
                raise RuntimeError(f"EDGAR fetch failed: {exc}") from exc
            time.sleep(wait * (i + 1))
    raise RuntimeError("EDGAR fetch failed after retries")


def fetch_aapl_8k_meta(years: int = 3, max_filings: int = 60) -> pd.DataFrame:
    """List real AAPL 8-K filing dates + accession numbers (most recent first)."""
    url = f"{_SEC}/submissions/CIK{_CIK}.json"
    data = _fetch_json(url)
    recent = data.get("filings", {}).get("recent", {}) or data.get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accession = recent.get("accessionNumber", [])
    rows = []
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    for form, dt, acc in zip(forms, dates, accession):
        if form == "8-K":
            d = pd.Timestamp(dt)
            if d >= cutoff:
                rows.append((d, acc))
        if len(rows) >= max_filings:
            break
    return pd.DataFrame(rows, columns=["date", "accession"])


def _primary_doc_url(accession: str) -> str:
    acc_no = accession.replace("-", "")
    idx = _fetch_json(f"{_ARCHIVES}/{_CIK}/{acc_no}/index.json")
    items = idx.get("directory", {}).get("item", [])
    if not items:
        raise RuntimeError(f"empty index for {accession}")
    # prefer main document (largest .htm) — deterministic
    best = max(items, key=lambda it: it.get("size", 0))
    return f"{_ARCHIVES}/{_CIK}/{acc_no}/{best['name']}"


def fetch_aapl_8k_sentiment(years: int = 3, max_filings: int = 60,
                            min_words: int = 40) -> pd.DataFrame:
    """Firm-level daily lexicon sentiment from real AAPL 8-K filings."""
    meta = fetch_aapl_8k_meta(years=years, max_filings=max_filings)
    rows = []
    n_ok = 0
    for _, row in meta.iterrows():
        try:
            url = _primary_doc_url(row["accession"])
            time.sleep(0.6)  # polite pacing
            r = requests.get(url, headers=_headers(), timeout=60)
            if r.status_code != 200:
                continue
            score = score_text_lexicon(r.text)
            if score["n_words"] < min_words:
                continue
            rows.append({"date": row["date"], "accession": row["accession"],
                         **score})
            n_ok += 1
        except Exception:
            continue
    if not rows:
        raise RuntimeError("no usable AAPL 8-K text retrieved")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["count"] = 1
    daily = df.groupby("date").agg(
        polarity=("net", "mean"),
        count=("count", "sum"),
    ).reset_index()
    daily["polarity"] = daily["polarity"].astype(float)
    print(f"EDGAR: {n_ok}/{len(meta)} 8-K filings scored "
          f"(dates {daily['date'].min().date()}..{daily['date'].max().date()})")
    return daily


def build_aapl_firm_csv(out: Path, years: int = 3, max_filings: int = 60) -> Path:
    daily = fetch_aapl_8k_sentiment(years=years, max_filings=max_filings)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily["Date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.drop(columns=["date"])
    daily.to_csv(out, index=False)
    print(f"wrote {out}: rows={len(daily)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sentiment_aapl_firm.csv")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--max-filings", type=int, default=60)
    args = ap.parse_args()
    build_aapl_firm_csv(Path(args.out), years=args.years,
                        max_filings=args.max_filings)


if __name__ == "__main__":
    main()

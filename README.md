# Improving GRU Sentiment Stock Forecasting

Research project on improving GRU-based stock forecasting by incorporating sentiment signals.

> Project mentor: Padmakumari Ma'am

## Reference paper

> Padmakumari P., Pranav R. Swaminathan, Harshavardhan S.R., Viswadarsini S., Jaiaakaash J.S. (2026). **Precision Investing: Combining Gated Recurrent Unit and Sentiment Analysis for Enhanced Stock Market Predictions.** *Computational Economics*. DOI: [10.1007/s10614-026-11393-w](https://doi.org/10.1007/s10614-026-11393-w)

This project builds on that paper: our baseline mirrors its protocol (lookback 30, chronological 70/15/15 split, forward-fill imputation, MinMax scaling, Adam + MSE + early stopping, close-price-only features) and adds leakage-free evaluation plus sentiment-augmented extensions. Protocol details and our alignment/gaps are tracked in `NOTES.md`.

## What is implemented and tested

- **Leakage-free baseline** — `src/baseline.py`: chronological sorting, past-only missing-value handling (forward-fill), scaler fit on the training split only, sliding windows; naive persistence vs. GRU/LSTM (PyTorch); metrics in original price units (MAE, RMSE, MAPE, R², directional accuracy), saved as JSON/CSV/PNG under `results/`.
- **Experiment harness** — `src/experiment.py`: expanding/rolling walk-forward with per-fold fitting, and sentiment ablation on identical splits/seeds (`--mode sentiment-ablation`). When no real sentiment columns exist, `--mode sentiment-ablation` refuses cleanly; `--mode sentiment-ablation-stock-only` runs a **zero-feature dummy control** (price-only vs. a constant-zero sentiment column) — it is not a true sentiment ablation.
- **Return / directional diagnostics** — `src/return_experiment.py`: dead-zone directional accuracy (`|r| > ε`), variance-standardized targets (`logvol`/`pctvol`), and an optional ridge baseline on the same leak-free splits.
- **Real sentiment ingestion** — `src/sentiment_gdelt.py`: fetches historical GDELT “Average Tone” (public API, no key), aligns to trading days, lags by one day (no look-ahead), and writes `data/sentiment_AAPL.csv` / `data/sentiment_005930.KS.csv` for true `--mode sentiment-ablation` runs.
- **Horizon / feature / model battery** — `src/horizon_experiment.py`: horizons 1/3/5d, causal technical + real sentiment transforms, and GRU vs Ridge vs HistGradientBoostingRegressor on identical leak-free splits (results in `results/hypothesis_battery.md`). Supports `--stride overlap|nonoverlap` for disjoint multi-day evaluation and reports a causal momentum baseline (results in `results/strict_verification.md`).
- **Real-data check (AAPL, 10y via yfinance):** baseline + 3-fold expanding walk-forward results, summarized in `NOTES.md` with artifacts in `results/`.

Quick runs:

```bash
python -m src.baseline --ticker AAPL --plot
python -m src.experiment --mode walk-forward --wf-type expanding --n-folds 3 --ticker AAPL
python -m src.experiment --mode sentiment-ablation --data data/prices.csv --sentiment-cols Sentiment
```

## Status

- [x] Repository scaffold
- [x] Leakage-free baseline (persistence / GRU / LSTM)
- [x] Experiment harness (walk-forward, sentiment ablation)
- [x] AAPL real-data baseline + walk-forward results
- [x] Real sentiment ingestion (GDELT tone; AAPL + Samsung) and true ablation runs
- [x] Dead-zone / vol-standardized / ridge diagnostics
- [x] Horizon (1/3/5d), causal-feature, and model-family hypothesis battery
- [ ] Sentiment source quality & coverage review (tone ≠ labelled news sentiment)
- [ ] Volume features; volatility-scaled dead-zone eps
- [ ] Return / directional-target variants (next step under discussion — see below)

## Repository layout

```
.
├── data/           # datasets (raw + processed)
├── src/            # source code
│   ├── baseline.py # baseline GRU/LSTM forecasting (leakage-free)
│   └── experiment.py # walk-forward + sentiment-ablation harness
├── experiments/    # experiment configs/notes
└── results/        # outputs, metrics, figures
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3-venv` is unavailable on your system (e.g., Debian/Ubuntu without it installed), you can instead install into your user site-packages:

```bash
pip install --user --break-system-packages -r requirements.txt
```

(`yfinance` is optional, only needed for `--ticker`.)

## Live Project Showcase

Static research showcase (verified numbers only) hosted on GitHub Pages:

**https://hudaifahwins.github.io/improving-gru-sentiment-stock-forecasting-padmakumari-maam/**

Source lives in `docs/` (`index.html`, `styles.css`, `app.js`, `.nojekyll`). Local preview:
`python3 -m http.server 8000 --directory docs`.

## Next step — question for Dr. Padmakumari P.

We want her guidance on **moving from raw price-level targets to return/directional targets** and **testing date-aligned sentiment features** on the same splits:

> Given our GRU beats persistence on directional accuracy but loses on RMSE/MAPE in raw price units, should we switch targets to returns/directional labels (with volatility-aware evaluation) and then run `--mode sentiment-ablation` with a real sentiment source — and what improvement over persistence would you expect a sentiment-augmented GRU to show?

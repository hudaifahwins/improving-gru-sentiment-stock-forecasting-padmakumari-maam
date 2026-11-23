# Improving GRU Sentiment Stock Forecasting

Research project on improving GRU-based stock forecasting by incorporating sentiment signals.

> Project mentor: Padmakumari Ma'am

## Reference paper

> Padmakumari P., et al. (2026). **Precision Investing: Combining Gated Recurrent Unit and Sentiment Analysis for Enhanced Stock Market Predictions.** *Computational Economics*. DOI: [10.1007/s10614-026-11393-w](https://doi.org/10.1007/s10614-026-11393-w)

This project builds on that paper: our baseline mirrors its GRU setup (lookback 30, hidden size 50) and adds leakage-free evaluation plus sentiment-augmented extensions.

## What is implemented and tested

- **Leakage-free baseline** — `src/baseline.py`: chronological sorting, past-only missing-value handling (forward-fill), scaler fit on the training split only, sliding windows; naive persistence vs. GRU/LSTM (PyTorch); metrics in original price units (MAE, RMSE, MAPE, R², directional accuracy), saved as JSON/CSV/PNG under `results/`.
- **Experiment harness** — `src/experiment.py`: expanding/rolling walk-forward with per-fold fitting, and sentiment ablation on identical splits/seeds (`--mode sentiment-ablation`).
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
- [ ] Sentiment dataset (news/headlines) and ablation on real data
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

## Next step — question for Dr. Padmakumari P.

We want her guidance on **moving from raw price-level targets to return/directional targets** and **testing date-aligned sentiment features** on the same splits:

> Given our GRU beats persistence on directional accuracy but loses on RMSE/MAPE in raw price units, should we switch targets to returns/directional labels (with volatility-aware evaluation) and then run `--mode sentiment-ablation` with a real sentiment source — and what improvement over persistence would you expect a sentiment-augmented GRU to show?

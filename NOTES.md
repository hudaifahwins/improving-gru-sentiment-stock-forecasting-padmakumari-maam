# Notes

Scratchpad for ideas, literature references, and decisions. Heavy experiments
and code live elsewhere; this file tracks the reasoning.

## Reference (verified from PDF, 2026-11-23)

> Padmakumari P., Pranav R. Swaminathan, Harshavardhan S.R., Viswadarsini S., Jaiaakaash J.S. (2026). **Precision Investing: Combining Gated Recurrent Unit and Sentiment Analysis for Enhanced Stock Market Predictions.** *Computational Economics*. DOI: 10.1007/s10614-026-11393-w. Received 27 Dec 2024 / accepted 22 Jun 2026.

## Paper protocol vs. our implementation

**Paper protocol (as published).**
- Data: Yahoo Finance, Samsung, daily, ~1250 rows; only Date + Close used; missing values filled from previous day (forward-fill).
- Scaling: MinMaxScaler (applied to the whole dataset). Windows: time steps of 30 (monthly).
- Split: chronological 70/15/15 (train/val/test); Adam, MSE loss, early stopping on validation loss, dropout on recurrent layers; batch size 64, 100 epochs (validation split also stated as 0.1 elsewhere).
- Models: GRU vs. MLS-LSTM (3 stacked LSTM layers, Keras/TensorFlow). GRU hidden size/layers are *not* reported.
- Metrics: R² (reported as % "accuracy"), MAPE, normalized RMSE (nRMSE = RMSE/(ymax − ymin)), adjusted R².
- Test results: GRU R²=0.948 (94.8%), MAPE=1.136%, nRMSE=0.048; MLS-LSTM R²=0.919 (91.9%), MAPE=1.380%, nRMSE=0.059.
- Sentiment: LSTM classifier (embedding → LSTM → dense + softmax; positive/neutral/negative), Kaggle news/tweet dataset (10,687 records; test accuracy 83.4%); daily scores aggregated into a numerical sentiment signal.
- Fusion: conflict-aware — agreeing price + sentiment signals → high-confidence prediction; conflicting signals → uncertainty flagged, not biased.

**Alignment with our code.**
- Matches: lookback 30 (`--lookback`), chronological 70/15/15, forward-fill imputation, MinMaxScaler, Adam + MSE, early stopping on val loss, close-price-only features, GRU/LSTM stacks.
- Improvements over the paper: scaler fitted on the training split only (paper scales the whole dataset — mild leakage); persistence baseline; expanding/rolling walk-forward with per-fold fitting; directional accuracy + MAE; deterministic seeds.
- Gaps to discuss with Dr. P.: (1) GRU hidden size/layers unreported (we use 50/1); (2) internal inconsistencies: dataset period (2016–2021 in §4/§5.1 vs. 2019–2024 in §5) and validation split (0.1 vs. 15%); (3) we don't yet compute nRMSE — adding it to `src/baseline.py` would allow head-to-head comparison with the paper's numbers; (4) the paper fuses sentiment post-hoc via a categorical signal rule, while our `--mode sentiment-ablation` concatenates daily numeric sentiment as a model feature — a different mechanism worth validating together.

## Open questions

- Which sentiment source? (news headlines, Twitter/X, Reddit, analyst reports)
- Sentiment granularity: daily aggregate vs. event-driven?
- Fusion strategy: feature concat, attention, or separate branch?

## Timeline

- 2026-11-23: repo scaffold + baseline code (GRU/LSTM, persistence) + experiment harness (walk-forward, sentiment ablation)
- 2026-11-23: real-data baseline check on AAPL (below)

## Real-data baseline check (2026-11-23, AAPL via yfinance)

**Setup.** Daily auto-adjusted close for AAPL, 2514 rows (10y), price-only (no sentiment columns in source). Lookback 30, GRU hidden 50 / 1 layer, Adam + MSE, early stopping on val loss, seed 42. Baseline: chronological 70/15/15 split. Walk-forward: expanding, 3 folds (fold size 628), per-fold scaler fit on that fold's training block only.

**Baseline (held-out test, ~15%).**

| model | MAE | RMSE | MAPE | R2 | DirAcc |
|---|---|---|---|---|---|
| persistence (y_hat=y_{t-1}) | 3.09 | 4.38 | 1.10% | 0.987 | 0.000 |
| GRU | 13.86 | 17.64 | 4.57% | 0.793 | 0.504 |

(GRU early-stopped at epoch 7; persistence DirAcc is ~0 by construction — it predicts zero change.)

**Walk-forward (expanding, 3 folds, mean ± std).**

| model | RMSE | MAPE | DirAcc |
|---|---|---|---|
| persistence | 3.04 ± 0.93 | 1.34% ± 0.14% | 0.002 ± 0.001 |
| GRU | 15.20 ± 7.69 | 6.50% ± 3.34% | 0.492 ± 0.030 |

**One question for Dr. Padmakumari P.** Given the plain price-level GRU loses to persistence on RMSE/MAPE (a common artefact of trending assets), should the project target *direction/profitability-aware* evaluation (e.g., sign/return-based targets, volatility-adjusted or log-return targets) rather than raw price levels — and if so, what baseline comparison would you expect for the GRU to be considered a meaningful improvement?

## Proposed next step (to bring to Dr. Padmakumari P.)

1. **Switch targets from raw close prices to returns / directional labels** (e.g., log-return or sign-of-return), with volatility-aware evaluation alongside RMSE/MAPE.
2. **Test date-aligned sentiment features** by running `--mode sentiment-ablation` on the same AAPL splits once we have a sentiment source (e.g., news headlines / analyst tone); the harness already supports this unchanged.


# Notes

Scratchpad for ideas, literature references, and decisions. Heavy experiments
and code live elsewhere; this file tracks the reasoning.

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


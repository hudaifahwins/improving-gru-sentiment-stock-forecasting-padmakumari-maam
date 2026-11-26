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
- Gaps to discuss with Dr. P.: (1) GRU hidden size/layers unreported (we use 50/1); (2) internal inconsistencies: dataset period (2016–2021 in §4/§5.1 vs. 2019–2024 in §5) and validation split (0.1 vs. 15%); (3) we now compute nRMSE in `compute_metrics` — the return-experiment report already prints it, so head-to-head comparison with the paper's numbers is possible (Samsung nRMSE 0.038 vs. paper 0.048); (4) the paper fuses sentiment post-hoc via a categorical signal rule, while our `--mode sentiment-ablation` concatenates daily numeric sentiment as a model feature — a different mechanism worth validating together.

## Open questions

- Which sentiment source? (news headlines, Twitter/X, Reddit, analyst reports)
- Sentiment granularity: daily aggregate vs. event-driven?
- Fusion strategy: feature concat, attention, or separate branch?

## Timeline

- 2026-11-23: repo scaffold + baseline code (GRU/LSTM, persistence) + experiment harness (walk-forward, sentiment ablation)
- 2026-11-23: real-data baseline check on AAPL (below)
- 2026-11-26: 1,423-paper method-driven literature expansion + gap audit (below)
- 2026-11-26: decisive 84-cell non-overlapping multi-day test (null; below)

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

## Latest verified additions (2026-11-23)

- **Dead-zone directional accuracy** (`|r| > 0.005`): AAPL mse/log DirAcc 0.489±0.003 → strict 0.461±0.002 (deadzone share 0.35); hybrid/log 0.496±0.021 → strict 0.498±0.042; logvol 0.482±0.005. Samsung mse/log 0.4214±0.0000 → strict 0.4161; hybrid 0.436±0.011; logvol 0.4214. Ridge baseline (same split): AAPL 0.4841, Samsung 0.4510. All near chance; no directional superiority claim.
- **Real sentiment ablation (GDELT “Average Tone”, lagged 1 day)**: AAPL Δ DirAcc −0.0022±0.0154 (≈ zero, within noise); Samsung Δ DirAcc 0.0000±0.0000. Tone adds no measurable directional signal on these stocks/periods.
- **Ingestion caveats**: GDELT DOC API is rate-limited (HTTP 429) and intermittently unavailable; retries with sleep succeeded after 2–4 attempts. Tone is an aggregate news-tone proxy, not labelled sentiment — coverage/quality still to be reviewed.
- Files: `data/sentiment_AAPL.csv` (1222 rows), `data/sentiment_005930.KS.csv` (1187 rows) — gitignored.

## Hypothesis battery (2026-11-23, real data, seeds 42/43/44)

`src/horizon_experiment.py`: horizons 1/3/5, causal features, real GDELT sentiment
transforms, GRU (mse) vs Ridge vs HistGradientBoostingRegressor on identical splits.

| Tick | Setup | DirAcc (eps=0) | DirAcc strict (0.005) | GRU price MAE |
|---|---|---|---|---|
| AAPL | price, h1 | 0.4832 ± 0.0089 | 0.4741 ± 0.0076 | 3.103 |
| AAPL | price+tech, h1 | 0.4680 ± 0.0109 | 0.4459 ± 0.0133 | 3.525 |
| AAPL | price+tech+sentiment, h1 | 0.5067 ± 0.0303 | 0.5061 ± 0.0494 | 10.842 |
| AAPL | price, h3 | 0.7762 ± 0.0118 | 0.8351 ± 0.0114 | 4.273 |
| AAPL | price, h5 | 0.7839 ± 0.0265 | 0.8120 ± 0.0286 | 6.910 |
| Samsung | price, h1 | 0.4224 ± 0.0014 | 0.4207 ± 0.0066 | 6775 |
| Samsung | price+tech, h1 | 0.4481 ± 0.0099 | 0.4531 ± 0.0173 | 8250 |
| Samsung | price+tech+sentiment, h1 | 0.4920 ± 0.0213 | 0.5103 ± 0.0238 | 29470 |
| Samsung | price, h3 | 0.6538 ± 0.0061 | 0.6730 ± 0.0052 | 47450 |
| Samsung | price, h5 | 0.6785 ± 0.0028 | 0.6842 ± 0.0025 | 103778 |
| AAPL | ridge baseline (h1, price) | 0.4832 ± 0.0089 | — | — |
| AAPL | gbr baseline (h1, price) | 0.4832 ± 0.0089 | — | — |
| Samsung | ridge baseline (h1, price) | 0.4224 ± 0.0014 | — | — |
| Samsung | gbr baseline (h1, price) | 0.4224 ± 0.0014 | — | — |

Honest reading:
- Multi-day (h=3/5) DirAcc is much higher — but targets overlap (P_{t+h} vs P_t),
  so these are NOT independent predictions; do not read them as tradeable signal.
- Causal tech features did not help at h=1 (AAPL worse, Samsung similar).
- Real sentiment transforms give noisy, statistically inconclusive deltas on both
  tickers (overlapping ± intervals with the price baseline).
- Ridge/GBR ≈ GRU on identical splits (no family beat another).
- Volume features not included (loader drops OHLCV); volatility-scaled dead-zone
  eps not yet run. Both are open limitations.

## Strict verification (fresh runs, commit 22e85ed+, seeds 42/43/44)

Full table in `results/strict_verification.md` (non-overlap = stride=horizon, disjoint windows).
Key results:

| Tick | Config | DirAcc (overlap) | DirAcc (non-overlap) | momentum DirAcc |
|---|---|---|---|---|
| AAPL | h1 price | 0.4832 ± 0.0089 | 0.4832 ± 0.0089 | 0.4841 ± 0.0000 |
| AAPL | h3 price | 0.7762 ± 0.0118 | 0.6494 ± 0.0347 | 0.5086 ± 0.0000 |
| AAPL | h5 price | 0.7839 ± 0.0265 | 0.7333 ± 0.0526 | 0.5000 ± 0.0000 |
| Samsung | h1 price | 0.4224 ± 0.0014 | 0.4224 ± 0.0014 | 0.4837 ± 0.0000 |
| Samsung | h3 price | 0.6538 ± 0.0061 | 0.6254 ± 0.0083 | 0.5044 ± 0.0000 |
| Samsung | h5 price | 0.6785 ± 0.0028 | 0.6422 ± 0.0302 | 0.5000 ± 0.0000 |

Strict conclusions:
- h=1 GRU ≈ momentum ≈ chance (AAPL 0.4832 vs 0.4841; Samsung 0.4224 vs 0.4837 — GRU below momentum on Samsung).
- Non-overlapping evaluation lowers multi-day DirAcc but h=3/5 still exceeds momentum (~0.50) and h=1; treated as diagnostic only (small disjoint test samples, trend autocorrelation).
- Ridge = GRU (0.4224 Samsung, 0.4832 AAPL); feature additions remain within seed noise (strongest: Samsung sentiment 0.492 ± 0.021 overlapping price 0.422 ± 0.001 — not conclusive at n=3).

## Method-driven literature expansion (2026-11-26)

Fetched 1,423 papers (2015–2026, all with abstracts, 993 OA) via
`scripts/fetch_relevant_papers.py` — gap-targeted queries rather than citation
lineage. Audit: `scripts/analyze_relevant_corpus.py`; memo:
`docs/literature_expansion_2026-11-26.md`.

Grounded counts over the 1,423 abstracts (vs the 575 lineage corpus): returns
21.9% (14.3%), price-level 55.4% (48.3%), GRU 26.9% (13.0%), transformer 20.2%
(11.0%), significance 9.0% (3.5%), GDELT 10 (0), Korea 25 (2), FinBERT 138,
LLM sentiment 92, leakage 44. Still near-absent: non-overlapping evaluation
(3 hits, 2 genuine) and train-only scaling (3 hits, 2 genuine); only 6 abstracts
mention significance + walk-forward + costs together.

Closest protocols: KSE-100 multi-horizon evaluation (2026, 13 non-overlapping
folds, 5+2 bps, DM), regime-gated attention under leakage control (2026,
no learned model beats historical mean), leakage-free index protocol with
train-only scaling (2026, MAKE), and GPT-sentiment look-ahead bias (2023).
Korea comparators: LLM news tone → next-day Korean returns (2026) and Korean ML
return forecasting 1987–2018 (2023).

Caveats: abstract-level keyword presence only; rare-signal counts hand-checked;
query plan is weighted toward GRU/sentiment, so cross-corpus deltas are not
field-wide trends. The arXiv supplement run left
`literature_corpus/arxiv_2026-11-26/` empty (no records).

## Decisive multi-day test (2026-11-26)

`scripts/run_decisive_multiday.py` (+ `assemble_decisive.py`): 5 markets (AAPL,
MSFT, 005930.KS, ^NSEI, 000001.SS), h=1/3/5, **43–50 disjoint walk-forward
windows per cell** (expanded from 24 to sharpen power), direct h-day targets,
train-only scaling, 3 GRU seeds, 5 bps/side, DM vs historical mean, Holm across
84 cells. Findings: `docs/decisive_multiday_findings.md`; process:
`docs/methodology_and_process.md`.

- **0/84** cells with positive excess DirAcc survive Holm. The only two
  significant cells are *negative*: AAPL GRU h1 −0.204 (Holm 0.030) and ^NSEI
  momentum h5 −0.227 (Holm <0.001).
- GRU mean excess h1 −0.212, h3 −0.221, h5 −0.022; mean DirAcc 0.446 vs
  majority bases 0.60/0.64/0.54. Best cell overall: AAPL Ridge h5 +0.114,
  n=44, Holm 1.00.
- Ridge h5 mean +0.005 (2/15 positive overall); momentum negative at every
  horizon; mean seed spread 0.032.
- The earlier E1–E8 AAPL h3 0.8067 (n=50) did not replicate under this protocol;
  treat it as a base-rate/small-n artifact.

## Sentiment ablation (2026-11-26)

`scripts/run_sentiment_ablation.py` (+ `assemble_sentiment.py`): paired design
on the two GDELT-covered tickers, 39–50 disjoint windows, 30 cells, 12 paired
comparisons, Holm within each family.

- Tone has no unconditional content: Pearson |r| ≤ 0.043 (p ≥ 0.143); Spearman |ρ| ≤ 0.048 (p ≥ 0.097)
  (overlapping windows, so effectively weaker).
- Adding tone to the GRU: Korea +0.160/+0.143/+0.154 DirAcc (DM p ≈ 0.01) but
  Holm 0.080/0.149/0.087; US −0.085/−0.049/−0.049 with Holm 0.013/0.014/0.044.
  Mean Δ +0.046 is a mix of opposite signs by market.
- 1/30 cells has a positive excess surviving Holm (AAPL h1 price-only GRU
  +0.106, Holm 0.003), but see period sensitivity below.
- **Period sensitivity:** the same AAPL h1 GRU is +0.106 (Holm 0.003) over the
  2024–2026 sentiment window and −0.204 (Holm 0.030) over the longer 2023–2026
  window. A walk-forward result describes a window, not a model.

Limits: n≈43–50/cell (minimum detectable excess ≈ 10 pp at 80% power), test-set
majority base is an oracle, one fixed GRU config, no intraday data.

## Proposed next step (to bring to Dr. Padmakumari P.)

1. **Switch targets from raw close prices to returns / directional labels** (e.g., log-return or sign-of-return), with volatility-aware evaluation alongside RMSE/MAPE.
2. **Test date-aligned sentiment features** by running `--mode sentiment-ablation` on the same AAPL splits once we have a sentiment source (e.g., news headlines / analyst tone); the harness already supports this unchanged.


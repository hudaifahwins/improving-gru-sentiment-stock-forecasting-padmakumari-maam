# Methodology and Process — how every number in this project was produced

*Last updated: 2026-11-26. This document is the audit trail for the PIF dossier
(`docs/PIF_Q1_submission.md`). Every headline number is traceable to a script, a
command and an output file listed here.*

---

## 1. What the project is

The anchor paper is Padmakumari, P. et al. (2026), *Precision Investing:
Combining GRU and Sentiment Analysis*, **Computational Economics**,
DOI `10.1007/s10614-026-11393-w`. It reports GRU price forecasting with
R² = 0.948, MAPE = 1.136 %, nRMSE = 0.048.

This project does **not** propose a new architecture. It builds the missing
**evaluation layer**: if a GRU-plus-sentiment pipeline is claimed to be
profitable, what happens when you test that claim the way an out-of-sample
trading evaluation should be tested? The deliverable is (i) a pre-registered
protocol, (ii) a reproducible audit of the anchor design and its natural
variants, and (iii) a decision-ready PIF dossier for a Q1 submission.

Two evidence streams:

| Stream | Question | Script | Output |
|---|---|---|---|
| Literature audit | Is the evaluation gap real across the field? | `scripts/analyze_relevant_corpus.py` | `literature_corpus/relevant_2026-11-26/ANALYSIS*.{json,md}` |
| Decisive evaluation | Does the anchor design survive an honest backtest? | `scripts/run_decisive_multiday.py` | `results/decisive_multiday/` |
| Sentiment ablation | Does the sentiment arm add anything? | `scripts/run_sentiment_ablation.py` | `results/sentiment_ablation/` |

---

## 2. Data pipeline

### 2.1 Prices

- Source: Yahoo Finance via `yfinance` (`auto_adjust=True`, so splits and
  dividends are adjusted), 7 years requested, cached to
  `data/cache/{TICKER}.csv` on first use.
- Universe (5 markets, 3 continents): `AAPL`, `MSFT` (US),
  `005930.KS` (Samsung, Korea), `^NSEI` (Nifty 50, India),
  `000001.SS` (Shanghai Composite, China).
- Cached span: 2019-11 → 2026-11, ≈ 1 700 trading days per series.
- Cleaning: parse dates, sort, drop duplicate dates, keep `close` and `volume`.

### 2.2 News sentiment (GDELT)

- Source: GDELT DOC 2.1 `TimelineTone` (public API, no key), via
  `src/sentiment_gdelt.py`.
- Query terms: `AAPL`, `"Samsung Electronics"`.
- **Leakage control:** the tone for day *t−1* is the value used at day *t*
  (`shift(1)` applied at ingestion), so a decision made at the close of day *t*
  cannot see news published after that close.
- Merge: `scripts/build_price_sentiment_csv.py` inner-joins prices and tone on
  trading dates → `data/prices_sentiment_{AAPL,005930.KS}.csv`
  (1 212 and 1 178 rows, 2021-11 → 2026-11). Inner join means no future row can
  align with a past one.
- Only these two tickers have sentiment coverage; the ablation is therefore a
  two-market paired study, not a five-market one. This is stated as a
  limitation, not hidden.

### 2.3 Features (identical for every model in a given arm)

Six causal price features per day, computed only from data up to that day:

1. `log(close)`
2. daily log return
3. 5-day momentum (`log P_t − log P_{t−5}`)
4. 20-day momentum
5. 10-day rolling volatility of returns
6. log-volume change

The sentiment arm appends: 7. sentiment level (lagged tone), 8. 3-day mean
sentiment (rolling, `min_periods=1`, causal).

### 2.4 Scaling

Standardisation uses the **mean and standard deviation of the origin's history
only** (`feat[:origin]`), recomputed at every walk-forward origin. The test
window is transformed with those training statistics. This removes the most
common silent leak in GRU pipelines (fitting the scaler on the full sample).
Only 3 of 1 423 audited papers report doing this (see §5).

---

## 3. Evaluation protocol (pre-registered)

The protocol was fixed before the runs and is applied identically to every
model:

1. **Expanding walk-forward.** Evaluation starts at the 50 % point of each
   series. At each origin the model is trained on everything before the origin
   and predicts the next block.
2. **Disjoint evaluation windows.** Origins advance by exactly `h` days, and
   each origin evaluates one `h`-day block. No two evaluation windows overlap,
   so the effective sample size equals the nominal one. This fixes the
   overlapping-window inflation that makes daily-horizon accuracy look far more
   precise than it is.
3. **Direct multi-horizon target.** Target is the `h`-day log return
   `log(P_{t+h}/P_t)` with `h ∈ {1, 3, 5}`. The model is *not* asked for a
   1-step return that is then compounded `h` times; that mismatch (train 1-day,
   report h-day) is one of the audited failure modes.
4. **Seed discipline.** GRU is trained with 3 seeds (42, 123, 2026); the
   prediction is the seed average, and the across-seed standard deviation of
   directional accuracy is reported as `seed_spread`. A single-seed result is
   never reported as a finding.
5. **Costs and turnover.** Signals are long/short on the sign of the predicted
   return. Net P&L charges **5 bps per side** on every position change; turnover
   is reported alongside every net number, so a high-accuracy/high-turnover
   strategy cannot hide behind gross returns.
6. **Class-imbalance honesty.** Directional accuracy is compared with the
   *majority-class base rate of the same test windows*, not with 50 %.
   `excess = DirAcc − majority_base` is the quantity that matters.
7. **Benchmarks.** Historical mean (expanding), majority class, persistence
   (predict zero), causal momentum (`log P_t/P_{t−h}`), Ridge on flattened
   features, GRU. Every model sees the same origins, target and scaler rule.
8. **Statistics.** Diebold–Mariano with Newey–West HAC variance on squared
   errors vs the historical-mean forecast; Holm step-down correction across all
   cells in a family; McNemar exact test for paired direction calls in the
   sentiment ablation.
9. **Two test families, corrected separately.** (A) each arm vs historical mean
   — "does the model beat the mean?". (B) sentiment arm vs its price-only twin
   — "does sentiment add?".

### Model hyper-parameters (held fixed, not tuned per cell)

| Item | Value |
|---|---|
| Lookback window | 30 trading days |
| GRU | 1 layer, hidden 32, Adam lr 1e-3, MSE loss |
| Epochs | 15, early stop patience 8 on a 15 % tail validation split inside training data |
| Ridge | α = 1.0 |
| Seeds | 42, 123, 2026 |
| Cost | 5 bps/side = 0.0005 |
| Threads | `torch.set_num_threads(1)` per process |

No hyper-parameter was selected on the evaluation windows. That is a
deliberate conservative choice: the audit asks whether the *published recipe*
survives, not whether an optimally tuned variant can be found.

---

## 4. Experiment registry (exact commands)

```bash
# Literature audit (1 423 abstracts, keyword/theme signals)
python3 scripts/analyze_relevant_corpus.py

# Decisive multi-market evaluation — 5 tickers x h in {1,3,5}
# (run per ticker so parts execute in parallel; Holm is recomputed on the
#  merged set by the assembler, because correction must span all cells)
for t in AAPL MSFT 005930.KS '^NSEI' 000001.SS; do
  python3 scripts/run_decisive_multiday.py --tickers "$t" --horizons 1,3,5 \
      --origins 50 --epochs 15 --gru-tickers 'AAPL,005930.KS,^NSEI' \
      --outdir "results/decisive_multiday/parts/$(echo $t | tr -d '^' | tr . _)"
done
python3 scripts/assemble_decisive.py 'results/decisive_multiday/parts/*/summary.json' \
    results/decisive_multiday

# Sentiment ablation (2 sentiment-covered tickers, paired arms)
for t in AAPL 005930.KS; do
  python3 scripts/run_sentiment_ablation.py --tickers "$t" --horizons 1,3,5 \
      --origins 50 --epochs 15 --outdir "results/sentiment_ablation/parts/$t"
done
```

Outputs: `summary.json` (machine-readable, one record per cell) and
`findings.md` (tables) in each result directory. `results/` is gitignored; the
tables are copied into the tracked docs so the submission is self-contained.

### Runtime and hardware

- Machine: 8-core Linux, CPU-only PyTorch 2.15.0+cpu, Python 3.12.3.
- Five decisive parts + two sentiment parts run concurrently, each pinned to
  one thread.
- Determinism: seeds fixed for NumPy and Torch; single-threaded BLAS. Results
  are reproducible on the same library versions; exact bit-reproducibility
  across PyTorch versions is not claimed.

---

## 5. Literature audit (how the gap claim is measured)

Corpus: 1 423 papers collected 2026-11-26 into
`literature_corpus/relevant_2026-11-26/` (manifest + abstracts; 993 open
access). `scripts/analyze_relevant_corpus.py` tags each abstract with
theme/signal regexes and writes `ANALYSIS.json`, `ANALYSIS_TABLE.md` and
`docs/literature_expansion_2026-11-26.md`.

Headline counts (all re-verified against `ANALYSIS.json`):

| Signal | Papers | Share |
|---|---|---|
| Forecasts the price level (not the return) | 789 | 55.4 % |
| Forecasts returns | 312 | 21.9 % |
| GRU used | 383 | 26.9 % |
| Reports a significance test | 128 | 9.0 % |
| Uses GDELT as the text source | 10 | — |
| Non-overlapping evaluation windows | 3 (2 genuine after reading) | 0.2 % |
| Train-only scaler fitting | 3 (2 genuine) | 0.2 % |
| Korean market | 25 | — |
| India | 112 | — |

Method caveat: the tagging is regex-based on abstracts, so it is a *lower
bound* on each practice and can produce false positives (hence the manual read
of the non-overlap hits, which reduced 3 to 2). The claim we make is therefore
"the practice is rare in the visible literature", not "exactly N papers do X".

---

## 6. Verification log

What was actually run and checked (not assumed):

- `python3 -m py_compile` on every script added in this cycle.
- Every headline count re-read from `ANALYSIS.json` and from
  `results/*/summary.json`; percentages recomputed from raw counts.
- Literature corpus counts cross-checked against the manifest line count
  (1 423).
- The earlier `run_opensource_benchmark.py` artifact was found to be invalid
  (8 cells, single seed, n = 6, identical GRU arms, 1-step target scaled by h)
  and is **excluded** from every claim.
- No p-value in the docs is quoted without its cell, `n`, and correction family.

---

## 7. Threats to validity

1. **Sample.** 5 price series, 2 with sentiment; ~1 100 usable days each. This
   is enough to reject a large effect, not enough to certify a small one.
2. **Power.** `n ≈ 43–50` per cell after non-overlapping sampling. Minimum
   detectable excess directional accuracy at 80 % power is roughly 10
   percentage points; effects below that remain untested.
3. **Model class.** One GRU configuration, fixed a priori. A tuned or
   transformer-based model could differ; that is a follow-up, not a refutation.
4. **Sentiment source.** GDELT tone, not FinBERT embeddings. The anchor paper's
   text pipeline is not public, so we test the *design* (price + lagged tone),
   not their exact embeddings.
5. **Costs.** 5 bps/side is a liquid-equity assumption; small caps and Korea
   retail frictions are higher.
6. **Publication bias in the corpus.** Abstract-level tagging cannot see
   appendix-level protocol details.

---

## 8. Claim → evidence map

| Claim in the PIF | Evidence |
|---|---|
| "55 % of the field forecasts price level" | `ANALYSIS.json → signals.target_price_level = 789/1423` |
| "≤ 3 of 1 423 papers use non-overlapping windows" | `ANALYSIS.json → signals.non_overlapping = 3`, manual read → 2 |
| "0/84 cells beat the historical mean after Holm" | `results/decisive_multiday/summary.json` (50-origin run, §9) |
| "2 significant cells, both negative" | same file, `cells[].p_holm < 0.05` |
| Sentiment gain/loss figures | `results/sentiment_ablation/summary.json → paired[]` |
| "Sentiment has no unconditional content" | same file, `sentiment_ic[]` |
| Impact factors | JCR 2025 values listed in `docs/PIF_Q1_submission.md §H` |

---

## 9. Headline results of the final runs

Full tables: `docs/decisive_multiday_findings.md`. Figures:
`docs/assets/fig_excess_diracc.png`, `fig_net_vs_turnover.png`,
`fig_sentiment_delta.png`.

### 9.1 Decisive price-only test (84 cells, 43–50 disjoint windows each)

| Quantity | Result |
|---|---|
| Cells with positive excess surviving Holm | **0 / 84** |
| Cells significant after Holm | **2 / 84, both negative** (AAPL GRU h=1 −0.204, Holm 0.030; ^NSEI momentum h=5 −0.227, Holm <0.001) |
| GRU mean excess, h=1 / 3 / 5 | −0.212 / −0.221 / −0.022 |
| Ridge mean excess, h=1 / 3 / 5 | −0.090 / −0.153 / +0.005 |
| Momentum mean excess, h=1 / 3 / 5 | −0.099 / −0.144 / −0.055 |
| GRU mean DirAcc vs majority base | 0.446 vs 0.60 / 0.64 / 0.54 |
| Best cell (any model) | AAPL Ridge h=5, +0.114, n=44, Holm 1.00 |
| Mean GRU seed spread | 0.032 |
| Mean turnover (position changes/window) | GRU 0.46, Ridge 0.94, momentum 0.97 |

### 9.2 Sentiment ablation (30 cells, 12 paired comparisons)

| Quantity | Result |
|---|---|
| Tone vs next h-day return | Pearson \|r\| ≤ 0.043 (p ≥ 0.143); Spearman \|ρ\| ≤ 0.048 (p ≥ 0.097); overlapping, so effectively weaker |
| Positive excess surviving Holm | 1 / 30 (AAPL h=1 price-only GRU +0.106, Holm 0.003) |
| GRU+sent − GRU price, Korea (h=1/3/5) | +0.160 / +0.143 / +0.154; Holm 0.080 / 0.149 / 0.087 |
| GRU+sent − GRU price, US (h=1/3/5) | −0.085 / −0.049 / −0.049; Holm 0.013 / 0.014 / 0.044 |
| Ridge+sent − Ridge price | −0.10 to +0.11, no cell significant after Holm |

### 9.3 Period sensitivity

The same specification (AAPL h=1, price-only GRU, seeds 42/123/2026) gives
**+0.106 excess (Holm 0.003)** over the sentiment subsample's window
(≈2024-05 → 2026-11) and **−0.204 (Holm 0.030)** over the longer window
(≈2023-05 → 2026-11). Reported because it is the clearest illustration that a
walk-forward result is a statement about a window, not about a model.

---

## 10. What this means for the paper

The contribution is not "GRU fails". It is:

1. a **protocol** that makes GRU-plus-sentiment claims falsifiable
   (disjoint windows, direct h-day targets, train-only scaling, costs,
   seed averaging, Holm-corrected DM);
2. **evidence** about what happens when the protocol is applied to the anchor
   design and its variants across five markets;
3. a **reporting checklist** that lets a Q1 reviewer verify a forecasting paper
   in minutes.

That is a publishable, honest, and genuinely additive extension of the anchor
paper — and it is the version of the work that survives replication.

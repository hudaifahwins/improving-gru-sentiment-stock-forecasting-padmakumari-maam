# harshitt13/Stock-Market-Prediction-Model

Source: https://github.com/harshitt13/Stock-Market-Prediction-Model
Stars: 4
Updated: 2026-10-18T06:35:28Z

---

# Hybrid Stock Market Prediction Model

A walk-forward evaluation framework for next-day equity return forecasting,
comparing a tree ensemble, a bidirectional LSTM, a time-series transformer and
a stacked meta-ensemble against five baselines.

**The headline result is a null.** Across 30 US large caps and 2010–2026, none
of the models forecasts next-day returns better than trivial baselines. No
model achieves positive out-of-sample R², none beats its majority class, and
none has statistically significant alpha after correcting for multiple testing.

That is the finding, not a failure of the implementation. The value of this
repository is the evaluation protocol that makes such a result legible — and
the documented case study of what the same models reported before that protocol
existed.

---

## 1. The headline number

An earlier version of this project reported R² = 0.9627 and MAPE = 9.89% and
described itself as achieving state-of-the-art precision. Those numbers were
real arithmetic on a broken target.

| | Before (price target) | After (return target) |
|---|---|---|
| Target | next-day **close price** | next-day **log return** |
| Best model, primary metric | R² = **0.9627** | R²_OOS = **+0.0011** (fold median; mean −0.0022 ± 0.0109, pooled −0.0003) |
| Naive / zero-return baseline | R² = **0.9992**, MAPE **1.23%** | R²_OOS = **−0.0023** (fold median; pooled −0.0028) |
| Does the best model beat the baseline? | **No** — and the metric hid it | **Not distinguishably** — a thousandth above a benchmark at zero, with a fold sd of 0.011 and DM p = 0.47 |

Both runs use *identical frozen data* (`docs/frozen_aapl_raw.csv`) and an
identical 12-fold configuration.

The old model was asked to predict tomorrow's price while being given today's
price as a feature. It learned the identity function. On a trending price
series, R² measures the price series — so the naive "tomorrow equals today"
baseline scored **higher than every model** (R² 0.9992, MAPE 1.23%) while the
best model managed 0.9627 and 9.89%. Every model was worse than doing nothing,
and the metric could not show it.

Full tables: [`docs/baseline_before_refactor.csv`](docs/baseline_before_refactor.csv),
[`docs/baseline_after_refactor.csv`](docs/baseline_after_refactor.csv).

---

## 2. What this project does **not** claim

This is a null result on a narrow question. It is not any of the following.

**It does not claim markets are efficient.** Efficiency is a theory about all
information and all horizons. This tests 31 technical and macro features on one
horizon.

**It does not claim equity returns are unpredictable.** A large literature
finds predictability at monthly and annual horizons, in the cross-section
rather than the time series, and conditional on variables absent here
(valuation ratios, order flow, earnings revisions, sentiment, options-implied
measures). Nothing here speaks to any of that.

**It does not claim deep learning cannot work in finance.** Two architectures,
one hyperparameter configuration each, one seed for the headline and five for
the variance study. A negative result at this scale is evidence about *these
runs*, not about a model class.

**It does not claim these models are correctly specified.** The neural models
were given a 90-day lookback and default-ish hyperparameters. Optuna search was
available but disabled for the reported runs. A better-tuned model might do
better; this repository does not test that.

The claim is narrow and specific: **with these features, this horizon, this
universe and this protocol, the models do not beat trivial baselines, and the
protocol is strict enough that had they done so, the result would be
credible.**

---

## 3. Results

### 3.1 Table 1 — AAPL headline

AAPL, 2010-03-16 → 2026-09-03, 12 expanding walk-forward folds, common
evaluation window of 2520 forecast days (2016-05-27 → 2026-06-05). R²_OOS is
reported three ways from the same per-fold series: the **fold median** is the
headline, the fold mean ± sd shows the spread, and the pooled value (a
variance-weighted ratio that the worst fold dominates; see section 8.3) is
secondary. `results/fold_aggregates_headline.csv`,
`python analysis/fold_aggregates.py`.

| Model | DA (%) | Majority (%) | DA − majority (pp) | PT p | R²_OOS, fold median | R²_OOS, fold mean ± sd | R²_OOS, pooled | RMSE (bps) |
|---|---|---|---|---|---|---|---|---|
| Tree Ensemble | 52.82 | 54.01 | −1.19 | 0.0586 | **−0.0081** | −0.0152 ± 0.0326 | −0.02037 | 183.4 |
| BiLSTM ᵃ | 52.47 | 54.01 | −1.54 | 0.7159 | **−0.0009** | −0.0043 ± 0.0113 | −0.00499 | 182.0 |
| Transformer ᵃ | 52.60 | 54.01 | −1.41 | 0.7065 | **−0.0003** | +0.0011 ± 0.0074 | +0.00102 | 181.5 |
| Ridge (returns) | 53.28 | 54.01 | −0.73 | 0.2698 | **0.0000** | +0.0004 ± 0.0033 | +0.00124 | 181.5 |
| Logistic (direction) | 53.88 | 54.01 | −0.13 | 0.5217 | **−0.0004** | −0.0008 ± 0.0027 | −0.00027 | 181.6 |
| Hybrid meta (+VIX) ᵇ | 52.65 | 54.01 | −1.37 | 0.7480 | **+0.0011** | −0.0022 ± 0.0109 | −0.00029 | 181.6 |
| Hybrid meta (no VIX) ᵇ | 52.30 | 54.01 | −1.71 | 0.5926 | **+0.0010** | −0.0016 ± 0.0120 | −0.00197 | 181.8 |
| Zero return | 45.99 | 54.01 | −8.02 | — | **−0.0023** | −0.0037 ± 0.0063 | −0.00276 | 181.8 |
| Historical mean | 54.01 | 54.01 | 0.00 | — | **0.0000** | 0.0000 ± 0.0000 | 0.00000 | 181.6 |
| AR(1) returns | 53.71 | 54.01 | −0.30 | 0.2355 | **−0.0027** | −0.0028 ± 0.0063 | −0.00347 | 181.9 |
| ARIMA(5,0,0) returns | 52.05 | 54.01 | −1.96 | 0.8962 | **−0.0082** | −0.0066 ± 0.0101 | −0.00753 | 182.3 |
| Random sign | 50.26 | 54.01 | −3.75 | 0.2745 | **−0.8630** | −1.1199 ± 0.5787 | −0.89480 | 249.9 |

> **ᵃ Single seed (42) drawn from a distribution measured over 5 seeds.**
>
> | | DA (%) | R²_OOS | PT p |
> |---|---|---|---|
> | BiLSTM range | 51.46 – 53.91 | −0.0102 – −0.0032 | 0.014 – 0.893 |
> | Transformer range | 52.27 – 52.95 | −0.0106 – **+0.0010** | 0.201 – 0.696 |
>
> Under the pooled aggregation the Transformer's +0.00102 was the only positive
> R²_OOS; under the fold median it is −0.0003 and the two meta rows are
> +0.0011 and +0.0010. Which row sits a thousandth above zero depends on the
> aggregation, and every one of them is inside a seed range or a fold sd that
> **crosses zero**. None is a positive result.
>
> **ᵇ Single seed, range not measured.** The meta-learner's inputs are the
> base-model predictions, so it inherits their seed variance and adds the
> meta-fit on top. Its true spread is at least as wide as the base models' and
> was not quantified.

**Not one model beats the majority class.** Always predicting "up" scores
54.01%; the best model reaches 54.04% on the fixture and 52.82% here.
Directional accuracy against 50% is meaningless on a series that rises 54% of
the time.

**No Diebold-Mariano test against the zero-return baseline is significant** on
2520 shared days — the smallest p-value is 0.177 (Transformer). `Random sign`
is significant at p<0.0001 *in favour of the baseline*, which is the sanity
check behaving.

### 3.2 The 30-ticker sweep

30 US large caps across sectors, same 12-fold configuration, seed 42.
595,224 predictions persisted to `results/predictions/`.

Tree Ensemble across all 30:

| | mean | median | sd | min | max |
|---|---|---|---|---|---|
| DA − majority (pp) | −2.17 | −2.41 | 1.33 | −4.16 | **+0.15** |
| R²_OOS, pooled | −0.116 | −0.096 | 0.074 | −0.343 | **−0.032** |
| **R²_OOS, fold median** | **−0.047** | −0.042 | 0.021 | −0.107 | **−0.014** |
| PT p-value | 0.440 | 0.408 | 0.312 | 0.021 | 0.932 |
| t(alpha) | −0.84 | −0.64 | 1.12 | −2.64 | +0.99 |

| | count |
|---|---|
| Tickers with R²_OOS > 0 | **0 / 30** — pooled, fold mean and fold median alike (`results/fold_aggregates_sweep.csv`) |
| Tickers with t(alpha) > 1.96 | **0 / 30** |
| Tickers beating majority class | 3 / 30 (by ≤ 0.15pp) |

**Does any ticker break the null?**

| test | raw p<0.05 | expected by chance | Holm-corrected | P(≥ k ‖ null) |
|---|---|---|---|---|
| Pesaran-Timmermann | 2 / 30 (AVGO, META) | 1.5 | **0 / 30** | 0.446 |
| alpha | 8 / 30 | 1.5 | **0 / 30** | <0.001 |

The two linear comparators (`src/linear_models.py`; `results/comparators/`)
on the same thirty tickers: R²_OOS within 0.017 of zero by fold median on
every ticker, positive on 12 (ridge) and 18 (logistic) of 30, which is what a
forecast of the historical mean scores; DA − majority mean −0.48 and −0.28 pp;
PT p < 0.05 on 5 of 28 and 4 of 28 defined tests (four comparator runs predicted
up every day and have no test). The binomial reference assumes independent tests; a
simulated null of equicorrelated statistics (`analysis/correlated_hit_null.py`,
`results/hit_count_null.csv`) gives P = 0.012 / 0.084 / 0.099 for the ridge's 5 at
ρ = 0 / 0.3 / 0.5 and 0.024 at the correlation of the PT statistics measured by block
bootstrap (0.04; the returns themselves correlate 0.36, but the statistic nets the
market component out); 0.049 / 0.123 / 0.128 / 0.062 for the logistic's 4; and for the
pooled 11 of 90, 0.006 independent, 0.126 at ρ_ticker 0.3 / ρ_model 0.8, 0.017 at the
measured 0.03 / 0.20. All eleven are positive-association hits (lower tail 0 of 86;
mean statistic +0.46, `results/mean_pt_null.csv`), but only 8 of the 11 beat their majority
rate pooled and 4 by fold median. Holm within model 0 and 1; Holm across all 90 direction
tests, 0; alpha hits 1+1 and 0+2 of 30 (negative + positive), none
surviving Holm. Paper §6.2 discusses the excess of uncorrected directional hits under both nulls; the one within-model Holm survivor (logistic, NKE, +2.14 pp pooled) has a fold-median edge of −0.66 pp.

Two PT hits against 1.5 expected is chance. The eight alpha hits look alarming
until you check the sign: **all eight are t < −1.96**, none positive. The tree
underperforms a rising market because it holds ~50% exposure at beta 0.44–0.65;
that is a shortfall, not skill.

### 3.3 Seed variance versus architecture difference

Five seeds each of the BiLSTM and Transformer on AAPL, same 12-fold
configuration.

The statistic is `mean|arch gap| / mean(seed sd)`. Its null distribution — both
architectures identical, five seeds each — was obtained by **simulation, not a
closed form** (400,000 trials of two iid normal samples at n=5):

```
null median        1.22
null 90% interval  [0.67, 1.77]
```

| metric | observed ratio | p (one-tailed) | inside 90% interval |
|---|---|---|---|
| R²_OOS | 1.66 | 0.083 | **yes** |
| Directional accuracy | 1.69 | 0.072 | **yes** |
| sd of predicted returns | 0.62 | 0.036 | no (lower tail) |

**The architecture difference is not distinguishable from seed noise.** Both
1.66 and 1.69 fall inside the interval you would get if the two architectures
were identical.

The 0.62 on prediction dispersion falls *below* the interval, which would
suggest reseeding moves that quantity more than switching architecture does.
**This is not claimed**: it is a single comparison at n=5, one of three
reported, and it is not corrected for that.

Two apparent positive findings dissolve across seeds:

- **Transformer R²_OOS changes sign** across seeds (−0.0106 to +0.0010).
- **BiLSTM seed 4 reports PT p = 0.0137**, the only configuration in this
  project beating its majority class with a significant PT statistic. Its
  other four seeds give 0.33, 0.52, 0.71, 0.89. The seed study ran **10 tests**
  (2 architectures × 5 seeds); the expected number of hits at p<0.05 is 0.5 and
  P(at least one) is **0.401**. This is the expected chance hit.

### 3.4 Economics

Long/flat on the predicted sign, 7.5bps round-trip, buy-and-hold on the same
2520 days.

| Model | Alpha ann. | t(alpha) | p Holm | Beta | Exposure | Sharpe net |
|---|---|---|---|---|---|---|
| Tree Ensemble | +0.0740 | 1.71 | **0.873** | 0.663 | 0.608 | 1.166 |
| Hybrid meta (+VIX) | −0.0419 | −1.88 | 0.667 | 0.937 | 0.888 | 0.861 |
| Transformer | +0.0156 | 0.39 | 1.000 | 0.742 | 0.845 | 0.963 |
| Ridge (returns) | +0.0390 | 1.23 | 1.000 | 0.860 | 0.846 | 1.115 |
| Logistic (direction) | −0.0162 | −1.05 | 1.000 

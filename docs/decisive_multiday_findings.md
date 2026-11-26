# Decisive multi-day test — findings (non-overlapping, cost-aware)

Run commands in `docs/methodology_and_process.md §4`. Outputs:
`results/decisive_multiday/summary.json`, `results/sentiment_ablation/summary.json`.

- 5 markets (AAPL, MSFT, 005930.KS, ^NSEI, 000001.SS) · horizons 1/3/5
- **43–50 disjoint evaluation windows per cell** (expanded from 24 to sharpen power)
- 3 GRU seeds averaged (42/123/2026), mean seed spread **0.032**
- 5 bps/side, turnover reported
- Diebold–Mariano (HAC) vs the historical-mean forecast, Holm across all 84 cells
- Direct h-day targets, train-only scaling, causal features

Figures: `docs/assets/fig_excess_diracc.png`,
`docs/assets/fig_net_vs_turnover.png`.

## Headline

**0 of 84 cells have a positive excess directional accuracy that survives Holm
correction.** Two cells are significant after correction, and both are *worse*
than the historical mean:

| cell | excess DirAcc | n | DM p | Holm |
|---|---|---|---|---|
| AAPL GRU h=1 | **−0.204** | 49 | 0.0004 | **0.030** |
| ^NSEI momentum h=5 | **−0.227** | 44 | <0.0001 | **<0.001** |

### Mean excess directional accuracy by family

| model | h=1 | h=3 | h=5 | positive cells | mean net 5bps |
|---|---|---|---|---|---|
| historical mean | 0.000 | −0.013 | −0.033 | 0/15 | +0.139 |
| momentum | −0.099 | −0.144 | −0.055 | 3/15 | −0.047 |
| Ridge | −0.090 | −0.153 | **+0.005** | 2/15 | +0.045 |
| GRU | −0.212 | −0.221 | −0.022 | 1/9 | −0.043 |

Absolute DirAcc: GRU 0.446 on average, Ridge 0.516, momentum 0.496 — against
majority-sign base rates of 0.60 (h=1), 0.64 (h=3), 0.54 (h=5). The base rate
absorbs most of what a 50 %-threshold accuracy metric would call skill.

### GRU, all cells

| ticker | h | n | DirAcc | majority | excess | net 5bps | turnover | DM p | Holm | seed sd |
|---|---|---|---|---|---|---|---|---|---|---|
| 005930.KS | 1 | 48 | 0.417 | 0.604 | −0.187 | −0.233 | 0.48 | 0.0130 | 1.000 | 0.055 |
| 005930.KS | 3 | 48 | 0.417 | 0.667 | −0.250 | +0.014 | 0.48 | 0.0717 | 1.000 | 0.055 |
| 005930.KS | 5 | 43 | 0.558 | 0.512 | **+0.047** | +0.086 | 0.44 | 0.1149 | 1.000 | 0.063 |
| AAPL | 1 | 49 | 0.429 | 0.633 | **−0.204** | −0.132 | 0.47 | 0.0004 | **0.030** | 0.019 |
| AAPL | 3 | 49 | 0.408 | 0.653 | −0.245 | −0.097 | 0.47 | 0.0024 | 0.196 | 0.020 |
| AAPL | 5 | 44 | 0.523 | 0.523 | +0.000 | −0.007 | 0.43 | 0.0267 | 1.000 | 0.027 |
| ^NSEI | 1 | 49 | 0.347 | 0.592 | −0.245 | +0.030 | 0.35 | 0.0303 | 1.000 | 0.016 |
| ^NSEI | 3 | 48 | 0.417 | 0.583 | −0.167 | −0.020 | 0.44 | 0.0155 | 1.000 | 0.016 |
| ^NSEI | 5 | 44 | 0.500 | 0.614 | −0.114 | −0.032 | 0.57 | 0.0686 | 1.000 | 0.018 |

Best cell in the whole table: AAPL Ridge h=5, excess **+0.114** (n=44, Holm
1.00); 000001.SS Ridge h=5 +0.093 (Holm 1.00) and AAPL momentum h=5 +0.091
(Holm 1.00) follow. At n≈44 an excess of ≈0.10 is within sampling noise,
and none survives correction.

### Interpretation

1. **Daily direction is at or below the base rate for every family.** GRU is
   significantly *worse* than the historical mean on AAPL h=1 in MSE terms.
2. **The h=5 "improvement" is not established.** GRU h=5 mean excess is −0.022
   with 1 of 3 markets positive; the largest single cell (+0.047) is smaller
   than one seed spread (0.063) and far below significance at n=43.
3. **Costs and drift.** Average net returns are positive for Ridge (+0.045) and
   the historical mean (+0.139) because long exposure in a rising sample earns
   the drift, not because direction is predicted. Turnover separates the arms:
   GRU 0.46 vs Ridge 0.94 vs momentum 0.97 position changes per window.
4. **The earlier batch E1–E8 AAPL h=3 = 0.8067 (n=50) does not replicate.**
   Under this protocol AAPL h=3 GRU is 0.408 vs a 0.653 base. The earlier figure
   is a base-rate / small-sample artifact.

---

## Sentiment ablation — the anchor paper's own design

Paired design on the two tickers with a lagged GDELT tone series
(`data/prices_sentiment_*.csv`, 2021-11 → 2026-11). Same protocol, 39–50
windows per cell, 30 cells, Holm within each test family.

### Unconditional sentiment content (all insignificant)

| ticker | h | n | Pearson r | p | Spearman r | p |
|---|---|---|---|---|---|---|
| 005930.KS | 1 | 1177 | −0.008 | 0.788 | −0.033 | 0.254 |
| 005930.KS | 3 | 1175 | −0.036 | 0.213 | −0.027 | 0.364 |
| 005930.KS | 5 | 1173 | −0.043 | 0.143 | −0.029 | 0.322 |
| AAPL | 1 | 1211 | −0.036 | 0.216 | −0.048 | 0.097 |
| AAPL | 3 | 1209 | −0.014 | 0.625 | −0.017 | 0.564 |
| AAPL | 5 | 1207 | −0.026 | 0.373 | −0.023 | 0.426 |

Every Pearson |r| ≤ 0.043 (p ≥ 0.143) and every Spearman |ρ| ≤ 0.048 (p ≥ 0.097), *before* any overlap adjustment
(these windows overlap, so the effective n is n/h and the true p-values are
larger). **Tone carries no detectable unconditional information about the next
1–5 day return in this sample.**

### Paired effect of adding sentiment to the GRU

Δ = DirAcc(GRU+sentiment) − DirAcc(GRU price-only), same origins, same seeds.

| ticker | h | n | Δ DirAcc | DM p | Holm | McNemar p |
|---|---|---|---|---|---|---|
| 005930.KS | 1 | 50 | **+0.160** | 0.0089 | 0.080 | 0.200 |
| 005930.KS | 3 | 49 | **+0.143** | 0.0213 | 0.149 | 0.265 |
| 005930.KS | 5 | 39 | **+0.154** | 0.0109 | 0.087 | 0.307 |
| AAPL | 1 | 47 | **−0.085** | 0.0011 | **0.013** | 0.597 |
| AAPL | 3 | 41 | **−0.049** | 0.0013 | **0.014** | 0.832 |
| AAPL | 5 | 41 | **−0.049** | 0.0044 | **0.044** | 0.824 |

- Korea: sentiment *helps* direction by 14–16 pp and the error reduction is
  nominally significant (p ≈ 0.01), but **none of the three cells survives Holm
  correction** (0.08–0.15).
- US: sentiment *hurts* by 5–8 pp, and this survives Holm at all three horizons.
- **The sign flips by market**, so the pooled mean (+0.046) is a mix of a large
  Korean gain and a smaller US loss. The honest conclusion: sentiment is not a
  reliable source of directional edge, and its effect is market-dependent.
- Ridge reacts differently (Samsung −0.10 at h=3/5, AAPL +0.11 at h=1); no Ridge
  paired cell is significant after Holm.

### Arm-level highlights

- 1 of 30 cells has a positive excess surviving Holm: **AAPL h=1 price-only GRU,
  +0.106 (DirAcc 0.660 vs 0.553 base, n=47, Holm 0.003)** — but see period
  sensitivity below.
- 4 of 30 are significantly *worse* than the historical mean, e.g. Samsung h=5
  price-only GRU −0.333 (n=39, Holm 0.046) and AAPL h=3 GRU+sentiment −0.220
  (Holm 0.008). DM significance here is about forecast error, so a cell can be
  significant with zero directional excess (AAPL h=5 GRU, excess 0.000,
  Holm 0.028) — direction and error are not the same claim.

### Period sensitivity (important caveat)

The sentiment sample starts later (2021-11) than the price cache (2019-11), so
its evaluation window is roughly 2024-05 → 2026-11 instead of 2023-05 → 2026-11.
The *same* AAPL h=1 price-only GRU is **−0.204 excess** over the longer window
(49 windows, Holm 0.030) and **+0.106** over the shorter one (47 windows, Holm
0.003). Both are "significant" in opposite directions. This is the single most
useful caution in the study: **a walk-forward result is a statement about a
window, not about a model.** Any paper reporting one without the other is
reporting a period artifact as skill.

---

## Limits

- 43–50 disjoint windows per cell; minimum detectable excess at 80 % power is
  roughly 10 pp, so smaller real effects remain untested.
- The majority-sign base rate is computed on the evaluation window itself
  (deliberately conservative / oracle).
- Five liquid markets, ~7 years, one fixed GRU configuration, no intraday data.
- Sentiment is GDELT tone, not the anchor paper's private embedding pipeline;
  we test the *design* (price + lagged tone), not their exact text model.
- No hyper-parameter tuning on the evaluation windows — by design, so
  "a better-tuned GRU could do better" remains an open follow-up, not a
  refutation.

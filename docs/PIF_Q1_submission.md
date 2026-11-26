# Paper Information Form (PIF) — Q1 submission dossier

**Working title:** *Beyond Price-Level Accuracy: A Leakage-Controlled, Base-Rate-Aware Evaluation of GRU–Sentiment Stock Forecasting Across Horizons and Markets*

**Anchor paper:** Padmakumari P., Swaminathan P.R., Harshavardhan S.R., Viswadarsini S., Jaiaakaash J.S. (2026). *Precision Investing: Combining Gated Recurrent Unit and Sentiment Analysis for Enhanced Stock Market Predictions.* Computational Economics. DOI 10.1007/s10614-026-11393-w.

**Positioning in one sentence:** We replicate and extend the anchor paper's GRU + sentiment design under a leakage-controlled protocol, and show which of its reported gains survive honest baselines, non-overlapping multi-day evaluation, transaction costs and statistical testing — a contribution the anchor paper's own literature (0/575 papers reporting non-overlapping evaluation) does not yet contain.

> **Note on this document:** sections A–L mirror the fields most institutional PIFs request (journal details, novelty, method, preliminary evidence, roles, timeline, recommendation). If your form has different fields, copy the matching section; every number below comes from verified runs of the committed harness in this repository (`NOTES.md` plus the gitignored `results/` artifacts) and is labelled *verified* or *planned*. Journal quartiles must be confirmed on the journal's current Scopus/JCR listing before submission — no impact factor is asserted here.

---

## A. Administrative details

| Field | Entry |
|---|---|
| Manuscript type | Original research article |
| Title | Beyond Price-Level Accuracy: A Leakage-Controlled, Base-Rate-Aware Evaluation of GRU–Sentiment Stock Forecasting Across Horizons and Markets |
| First author | [Student name], [Department], [Institution] |
| Senior / corresponding author | Dr. Padmakumari P., [Department], [Institution] |
| Co-authors | [Supervisor], [Collaborators] |
| Target journal | [Journal name] — see §I shortlist |
| ISSN / publisher | [Fill from journal page] |
| Quartile (to verify) | Q1 — confirm on Scopus/JCR listing for the current year |
| Indexing | Scopus, Web of Science (confirm for chosen title) |
| Data | Public: Yahoo Finance prices; GDELT tone; SEC EDGAR 8-K (AAPL). No personal data. |
| Code | Reproducible Python pipeline in this repository; seeds and CLIs documented |
| Funding | [Nil / grant number] |
| Conflicts | None declared |
| Ethics | Not applicable (public market and public-filing data) |

---

## B. Research summary (≈150 words)

Published GRU-based stock forecasting papers typically report impressive price-level accuracy but are rarely tested against naive baselines, under strictly chronological splits, with non-overlapping evaluation windows, or with statistical significance testing. We take a recent GRU + sentiment study as the anchor, reproduce its protocol faithfully, and then apply a leakage-controlled, base-rate-aware evaluation across horizons (1, 3, 5 days), five markets (US, Korea, India, China), multiple seeds, and a paired sentiment ablation. We report directional accuracy relative to the majority-sign base rate and a causal momentum baseline, Diebold–Mariano tests with Holm correction, and transaction-cost-adjusted backtests. The result is a reproducible benchmark that separates genuine predictive content from artifacts of scaling, overlapping windows and base rates — and a practical reporting checklist for financial ML papers. Both positive and null outcomes are informative: the protocol itself is the contribution.

---

## C. What the anchor paper establishes — and where it stops

All statements below are from the published paper as audited in `NOTES.md`.

| Anchor-paper element | As published | Consequence |
|---|---|---|
| Data | Samsung daily, ~1,250 rows, Date + Close only | Single market, single asset |
| Scaling | MinMaxScaler on the **whole dataset** | Mild look-ahead leakage |
| Split | Chronological 70/15/15 | Appropriate ordering |
| Targets | Raw closing price | Price-level accuracy; direction not isolated |
| Baselines | GRU vs MLS-LSTM only | No persistence / random-walk baseline |
| Metrics | R² (reported as "accuracy"), MAPE, nRMSE, adjusted R² | No directional accuracy, no significance tests |
| Sentiment | LSTM classifier on Kaggle news/tweets; categorical post-hoc fusion | Not date-aligned to the price series in a reproducible way |
| Reproducibility | GRU hidden size/layers unreported; dataset period and validation split inconsistent between sections | Hard to replicate exactly |

**None of this is a weakness of the science** — it reflects the conventions of the venue and the literature: in our 575-paper lineage corpus, only 3.5% of abstracts mention significance testing and **0/575 report non-overlapping multi-day evaluation**. The extension is therefore a natural next paper, not a correction.

---

## D. Added value: eight concrete contributions

Each contribution is tied to evidence already produced in this repository.

**C1 — A leakage-free replication of the anchor design.**
Same lookback (30), same chronological 70/15/15 logic, same GRU family, but the scaler is fit on training rows only and sentiment is lagged one trading day. *Why it matters:* isolates how much of the anchor result depends on whole-dataset scaling. *Evidence:* our harness reproduces the price-level pipeline and reports persistence alongside it (`NOTES.md`, AAPL baseline).

**C2 — Return and directional targets, not just price levels.**
We evaluate log-return and volatility-standardised targets with dead-zone directional accuracy (`|r| > 0.005`). *Why it matters:* price-level R² is dominated by trend and can look excellent while direction is at chance. *Evidence:* AAPL daily DirAcc 0.4832 ± 0.0089; Samsung 0.4224 ± 0.0014 — both at or below the majority-sign base rate.

**C3 — Non-overlapping multi-day evaluation.**
Disjoint h-step windows with a causal momentum baseline and reported n per horizon. *Why it matters:* overlapping P_{t+h} vs P_t targets are not independent predictions; the literature almost never checks this. *Evidence:* only **2 of 1,423** method-targeted papers genuinely evaluate disjoint folds (audited in `docs/literature_expansion_2026-11-26.md`).

**C4 — Base-rate-aware reporting.**
Every directional claim is paired with the majority-sign base rate, persistence and momentum. *Why it matters:* a "62% accuracy" on a 60%-up sample is not skill. *Evidence:* our batch audit shows several apparent wins collapse to the base rate once this is applied (`results/batch_e1e8/SUMMARY.md`).

**C5 — Statistical rigor.**
Diebold–Mariano tests with Newey–West/HAC standard errors, Holm correction across tested cells, and three seeds averaged with the seed spread reported. *Why it matters:* only 9.0% of the 1,423 expansion abstracts mention significance testing, and only 6 combine significance + walk-forward + costs.

**C6 — Five-market replication including Korea, India and China.**
AAPL, MSFT, Samsung (005930.KS), NIFTY 50 (^NSEI) and Shanghai Composite (000001.SS), all under one protocol. *Why it matters:* Korea appears in only 1.8% of the 1,423-paper expansion (0.3% of the 575-paper lineage corpus); the anchor paper's market is under-replicated, and no audited paper covers all five under identical rules.

**C7 — Honest sentiment ablation.**
Lagged GDELT average tone is ablated against identical price-only splits, paired by origin and seed. *Why it matters:* the anchor paper's fusion rule is not comparable to simple feature concatenation, so the incremental value of text must be measured, not assumed. *Evidence:* tone has no unconditional predictive content (Pearson |r| ≤ 0.043, p ≥ 0.143; Spearman |ρ| ≤ 0.048, p ≥ 0.097); adding it to the GRU moves Korean direction +0.14 to +0.16 (Holm 0.08–0.15) but US direction −0.05 to −0.09 (Holm 0.01–0.04). The gain is not robust; the loss is. No pooled "sentiment works" claim survives.

**C8 — A reporting checklist and open benchmark.**
A 19-item mandatory reporting checklist plus code: `docs/reporting_checklist.md` (split chronology, scaler-fit location, sentiment lag proof, window stride, base rate, correction family, seed spread, costs, per-market sentiment). *Why it matters:* recent reviews (e.g. Theodorakopoulos et al. 2026) explicitly call for auditable pipelines; the checklist is immediately reusable by the anchor paper's authors and by the journal's readers. Our own paper scores 19/19 plus every recommended item.

---

## E. Novelty relative to the field (grounded in the 1,423-paper audit)

| Dimension | Literature (1,423 abstracts) | This paper |
|---|---|---|
| Return targets | 21.9% | Core target set |
| Price-level targets | 55.4% | Compared, not relied on |
| Non-overlapping evaluation | 2 verified papers | Primary protocol |
| Train-only scaling | 2 verified papers | Enforced |
| Significance + walk-forward + costs together | 6 papers | Enforced |
| Korea coverage | 1.8% | One of five markets |
| India / China on identical rules | 7.9% / 8.5% | Both included |
| Sentiment | FinBERT 9.7%, GDELT 0.7% | Paired GDELT tone ablation (2 markets) |

The novelty is not a new architecture. It is the **evaluation layer**: a protocol that makes the anchor paper's claims testable, plus evidence about when they hold. That is a better Q1 fit than another incremental GRU variant, because the literature already has thousands of those.

---

## F. Method and pre-registered protocol

1. **Data.** Daily adjusted close, OHLCV where available; AAPL and 005930.KS; GDELT tone lagged one trading day; EDGAR 8-K sentiment for AAPL.
2. **Splits.** Chronological 70/15/15; expanding walk-forward with per-fold fitting; scaler fit on each training block only.
3. **Windows.** Lookback 30; horizons 1, 3, 5; **43–50 disjoint evaluation windows per cell** (window stride = h), so nominal and effective sample size coincide.
4. **Models.** Historical mean, majority sign, persistence, causal momentum, Ridge, GRU (3 seeds averaged); sentiment on/off in a paired ablation on the two sentiment-covered tickers.
5. **Metrics.** MAE, RMSE, nRMSE, MAPE, R²; directional accuracy with dead-zone ε; excess over majority-sign base rate.
6. **Inference.** Diebold–Mariano vs the historical-mean/persistence benchmark; Holm across families; 3 seeds averaged.
7. **Economics.** 5 bps per side; turnover; net return; disclosed n per cell.
8. **Pre-registration.** The decisive multi-day test and its success criterion are fixed before running: excess directional accuracy > 0 after costs, DM p < 0.05, same sign across markets. Full audit trail: `docs/methodology_and_process.md`.

---

## G. Evidence (final verified runs, reproducible)

| Result | Value | Source |
|---|---|---|
| AAPL held-out: persistence vs GRU RMSE | 4.38 vs 17.64 | `NOTES.md` |
| AAPL held-out: persistence vs GRU MAPE | 1.10% vs 4.57% | `NOTES.md` |
| Daily DirAcc, AAPL (price-only) | 0.4832 ± 0.0089 | strict verification |
| Daily DirAcc, Samsung (price-only) | 0.4224 ± 0.0014 | strict verification |
| Causal momentum DirAcc | 0.4841 (AAPL) / 0.4837 (Samsung) | strict verification |
| Samsung return-target nRMSE | 0.038 (anchor paper: 0.048) | `NOTES.md` |
| GDELT tone ablation, AAPL | Δ DirAcc −0.0022 ± 0.0154 | `NOTES.md` |
| Decisive test: cells with positive excess surviving Holm | **0 / 84** | `docs/decisive_multiday_findings.md` |
| Decisive test: significant cells, all negative | 2 / 84 — AAPL GRU h=1 −0.204 (Holm 0.030); ^NSEI momentum h=5 −0.227 (Holm <0.001) | same |
| Decisive test: GRU mean excess, h=1 / h=3 / h=5 | −0.212 / −0.221 / −0.022 | same |
| Decisive test: GRU mean DirAcc vs majority base | 0.446 vs 0.60 / 0.64 / 0.54 | same |
| Decisive test: best cell, any model | AAPL Ridge h=5 +0.114, n=44, Holm 1.00 | same |
| Sentiment content: tone vs next 1–5 day return | Pearson \|r\| ≤ 0.043 (p ≥ 0.143); Spearman \|ρ\| ≤ 0.048 (p ≥ 0.097) | `results/sentiment_ablation/summary.json` |
| Sentiment ablation: GRU+sent − GRU price, Korea | +0.160 / +0.143 / +0.154 (DM p ≈ 0.01) but Holm 0.080 / 0.149 / 0.087 | same |
| Sentiment ablation: GRU+sent − GRU price, US | −0.085 / −0.049 / −0.049, Holm 0.013 / 0.014 / 0.044 | same |
| Sentiment ablation: positive & Holm-significant cells | 1 / 30 (AAPL h=1 price-only GRU +0.106, Holm 0.003) | same |

**Honest reading:** daily direction is at or below the base rate for every model
family tested, and the decisive multi-day run does not rescue it: across 5
markets, 3 horizons and 84 cells, **no positive excess survives Holm
correction**. The earlier batch E1–E8 result (AAPL h=3 price+tech 0.8067, n=50)
did **not** replicate under direct h-day targets, train-only scaling and a
majority-sign base; it now reads as a base-rate/small-sample artifact.

The sentiment arm tells the same story with a twist that makes the paper
stronger rather than weaker: lagged GDELT tone has no unconditional predictive
content, and adding it to the GRU helps Samsung (+14 to +16 pp direction) while
*hurting* AAPL (−5 to −8 pp), with the Korean gain failing Holm correction and
the US loss passing it. A pooled "sentiment works" claim is therefore
unsupported — and the sign flip is exactly the kind of market-dependence the
audit is designed to expose.

**Period sensitivity** is reported rather than hidden: the same AAPL h=1
price-only GRU shows +0.106 excess (Holm 0.003) over the 2024–2026 sentiment
subsample and −0.204 (Holm 0.030) over the 2023–2026 window. Both are
"significant", in opposite directions, which is the paper's most practical
warning: a walk-forward number describes a window, not a model.

---

## H. Expected outcomes and claims

The decisive test returned the **null** result: 0/84 cells with positive excess survive Holm correction, and GRU is below the majority base at every horizon. The paper therefore reports that the anchor design's apparent multi-day skill does not survive base-rate and disjoint-window controls, and contributes:

1. the protocol (disjoint origins, direct h-day targets, train-only scaling, DM + Holm, costs);
2. the negative evidence across five markets and three horizons, with the two significant cells disclosed;
3. the paired sentiment ablation showing no reliable incremental value and a market-dependent sign;
4. the period-sensitivity demonstration (+0.106 vs −0.204 for the same model);
5. the reporting checklist;
6. the 1,423-paper gap audit showing why this evidence was missing.

This is a positive contribution: it is consistent with the 2026 leakage-controlled literature (arXiv:2606.29347 finds no learned model beats the historical mean) and gives the anchor paper's design its first properly powered replication. Null-result framing is explicit in the title and abstract, which protects against a "no novel model" objection.

---

## I. Q1 journal shortlist (verify current quartile before submission)

| Journal | Fit | Why |
|---|---|---|
| International Review of Financial Analysis | Finance + ML evidence | Strong fit for market/asset-pricing framing |
| International Journal of Forecasting | Evaluation methodology, horizons, benchmarks | Best scope match for a forecasting-evaluation paper |
| Expert Systems with Applications | Applied ML systems | Fit if the paper leans on the pipeline/checklist |
| Computational Economics | Anchor paper's venue | Continuity; lower IF, confirm scope/quartile |

### Impact-factor evidence (JCR 2025, released June 2026)

| Journal | 2025 JIF | JCR quartile | Evidence |
|---|---|---|---|
| International Review of Financial Analysis | **10.2** | Q1 (4/243, Business–Finance) | Elsevier journal page; JCR 2025 |
| Expert Systems with Applications | **9.4** | Q1 (confirm category) | Elsevier journal page |
| International Journal of Forecasting | **5.7** | Q1 (43/626 Economics; 90/426 Management) | JCR 2025 listing |
| Computational Economics | **2.1** (5-yr 2.3) | Confirm | Springer journal metrics |
| Journal of Forecasting | 2.6 | Q2 | Wiley journal metrics — **not** a Q1 target |

All candidates have a **positive impact factor**; the recommendation is a Q1 title with JIF ≥ 5.7. Cite the journal's own metrics page in the institutional form and attach the JCR screenshot, because third-party sites report different values (e.g. IJF appears as 5.7 on JCR and 6.46 on SciJournal).

Recommendation: lead with **International Review of Financial Analysis** (highest IF, finance audience) or **International Journal of Forecasting** (best scope); keep ESWA as the methods-heavy fallback. Do not submit to the anchor paper's venue first — it invites a direct "what is new versus our paper" objection that is easier to answer elsewhere.

---

## J. Collaboration plan (why Dr. Padmakumari should join)

| Role | Owner |
|---|---|
| Conceptualisation, anchor design | Dr. Padmakumari P. |
| Protocol, experiments, code | First author |
| Sentiment/data pipeline | First author + [collaborator] |
| Statistical analysis | First author; Dr. P. validates |
| Writing — original draft | First author |
| Writing — review, supervision, correspondence | Dr. Padmakumari P. |
| Funding / ethics / submission | Dr. Padmakumari P. |

**What she gains:** a rigorous extension of her paper with her as senior/corresponding author; a replication that strengthens rather than disputes her result; a Korea/US replication of her Samsung design; a reusable checklist; and a student who has already built and verified the pipeline. **What we ask:** protocol sign-off, two rounds of comments, and corresponding-author duties.

**Framing to use with her (important):** present this as *"your paper defined the design; we want to test how far it generalises under stricter controls"* — never as a critique. The four audit notes (hidden size, period/validation inconsistency, nRMSE, fusion mechanism) should be raised privately as questions for the extension, not listed as errors in the manuscript.

---

## K. Timeline (16 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Protocol freeze; Dr. P. sign-off; data refresh |
| 3–5 | Decisive multi-day run (5 tickers, 3 seeds, cost-aware) |
| 6–7 | Sentiment ablation replication; Korea extension |
| 8–9 | Statistical analysis; tables/figures; robustness |
| 10–11 | Draft manuscript; internal review |
| 12–13 | Co-author comments; revision |
| 14 | Formatting, reproducibility package, cover letter |
| 15–16 | Submission and response to desk review |

---

## L. Assessment and recommendation

| Criterion | Assessment |
|---|---|
| Gap is real and documented | Yes — 2/1,423 on non-overlapping evaluation; 0/575 in lineage corpus |
| Contribution beyond the anchor paper | Yes — evaluation layer, five markets, paired sentiment ablation, period sensitivity, checklist |
| Verified evidence exists | Yes — 84-cell decisive run, 30-cell sentiment ablation, period-sensitivity check |
| Feasibility | Yes — pipeline, data and harness already exist |
| Ethical / data risk | Low — public data, no personal data |
| Authorship clear | Yes — student first author, Dr. P. senior/corresponding |
| Journal fit | Q1, confirmed after quartile check |

**Recommendation: POSITIVE.** The manuscript is a methodologically grounded extension with verified preliminary results, a clear publication path whether the decisive test is positive or null, and a collaboration structure that protects the anchor paper's contribution.

---

## Appendix 1 — Cover note to Dr. Padmakumari (draft)

> Dear Dr. Padmakumari,
>
> I have been building on your 2026 Computational Economics paper on GRU + sentiment forecasting. I reproduced your design (lookback 30, chronological 70/15/15, GRU) and then tested it under a stricter protocol: train-only scaling, return/directional targets, non-overlapping multi-day windows, persistence and momentum baselines, Diebold–Mariano tests and transaction costs.
>
> The results are informative rather than contradictory. Your design's price-level accuracy replicates, but under the stricter protocol daily directional accuracy sits at the base rate for every model family, and the decisive test is now complete: 84 non-overlapping cells across five markets and three horizons, with no positive excess that survives Holm correction. The paired sentiment ablation is equally instructive — lagged GDELT tone helps Korean direction by 14–16 pp but hurts US direction by 5–8 pp, and the Korean gain does not survive correction. Your paper's design is the right vehicle to report this: it is the design the field actually uses.
>
> I have also checked where the literature stands: only 2 of 1,423 method-targeted papers evaluate non-overlapping windows, so this seems like a genuine gap your paper is well placed to fill.
>
> I would like to develop this into a Q1 submission with you as senior and corresponding author. The pipeline, data, full protocol and results are ready, and the audit trail is documented so every number can be checked. Would you be open to a short call this week?
>
> Best regards,
> [Name]

## Appendix 2 — Title options

1. *Beyond Price-Level Accuracy: A Leakage-Controlled, Base-Rate-Aware Evaluation of GRU–Sentiment Stock Forecasting* (recommended)
2. *Do GRU–Sentiment Models Beat Naive Baselines? A Multi-Horizon, Multi-Market Audit*
3. *Precision Investing Re-examined: Leakage Controls, Base Rates and the Limits of Daily Directional Prediction*

## Appendix 3 — Abstract draft (≈190 words)

GRU-based stock forecasting studies routinely report high price-level accuracy, yet such results are seldom tested against naive baselines, under strictly chronological splits, with non-overlapping evaluation windows, or with formal significance testing. We replicate a recent GRU + sentiment design and extend it with a leakage-controlled, base-rate-aware protocol across daily, three-day and five-day horizons in five markets (US, Korea, India, China). Using expanding walk-forward estimation, 43–50 disjoint windows per cell, train-only scaling, causal sentiment lags, persistence and momentum baselines, Diebold–Mariano tests with Holm correction, and transaction-cost-adjusted backtests, we separate predictive content from artifacts of scaling, overlapping targets and class imbalance. Across 84 non-overlapping cells, no model achieves a positive excess directional accuracy that survives Holm correction; the two significant cells are significantly worse than the historical mean. A paired ablation shows lagged news tone has no unconditional predictive content, helps Korean direction and hurts US direction, with opposite signs and no robust pooled gain. The same model is significantly positive over one window and significantly negative over another. We contribute a reproducible benchmark, an honest sentiment ablation, evidence on period sensitivity, and a reporting checklist for financial machine-learning papers.

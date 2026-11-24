# ITheClixs/Prediction-System-for-Istanbul-Stock-Exchange

Source: https://github.com/ITheClixs/Prediction-System-for-Istanbul-Stock-Exchange
Stars: 7
Updated: 2026-08-30T16:52:41Z

---

# Nothing Beats Zero, and the Standard Tests Cannot Tell: Simulation Calibration of Forecast Comparison on Correlated Equity Panels

**Abstract.** Short-horizon equity forecasting papers report a model that beats a benchmark. This one reports the apparatus, calibrates it against known truth, and finds three of its standard components badly wrong. On synthetic panels whose cross-sectional correlation, volatility and predictable share are fixed by construction and matched to a real panel, the two-sided Diebold-Mariano test applied to panel rows rejects a true null 22.84% of the time on four names correlated at 0.5697 and 79.13% on a hundred, against a nominal 5%. The inflation is not a finite-sample quirk: it has the closed form 2*Phi(-z/sqrt(1+(k-1)*rho-bar)), which the simulation tracks to within 0.0064 across a size range from 0.11 to 0.79. Holm's family-wise error rate follows it to 0.5875. Against a zero-forecast benchmark the models are nested, and the squared-error comparison declares a forecast that is pure estimation noise significantly *worse* than the benchmark in 92.19% of replications in which the benchmark is exactly correct, rising to certainty as the record lengthens. The False Strategy Theorem's threshold is cleared by roughly half of all skill-free grids at every level of dependence between trials, because it is an expectation and not a critical value. Three replacements are calibrated and hold their nominal level under heavy tails, clustered volatility, regime switches and record lengths from 60 to 500 sessions: aggregating loss differentials to one value per session, the Clark-West statistic, and a stationary bootstrap that resamples an entire configuration grid on a single index draw. The corrected apparatus is then applied to a fixed four-stock Borsa Istanbul prototype universe over 251 sessions, with an executable next-open-to-close target under a date-grouped purged walk-forward protocol. No forecast shows predictive content; Hansen's test of superior predictive ability does not reject (p = 0.6891); across a 72-configuration grid whose members are measured to have a mean pairwise correlation of 0.8395 and to behave like 3.77 independent trials, the best configuration in the grid returns 4.6259% net and carries an exact joint bootstrap p-value of 0.7715; and the zero-return null has the highest out-of-sample R-squared in all 72. What does not survive is the claim that 4 of the six fitted models are significantly worse than the null, which the calibration identifies as an artefact of fitting. Inverting the corrected tests, the smallest out-of-sample R-squared this design could separate from zero at 5% size and 80% power is 0.1132 in closed form and 0.2012 by simulation, against the 0.01 this literature treats as economically meaningful; reaching 0.01 would need 15,126 sessions. Widening the cross-section cannot substitute, because the loss differentials of correlated names carry at most 1.7663 independent rows per session here, however many names are added. And a closed-form feasibility bound shows the cost schedule requires a cross-sectional information coefficient of 0.3098 from any forecaster in this design, against the 0.0092 the portfolio model achieved per session. The negative result is therefore a statement about the experiment rather than about the market. The committed run replays byte-for-byte from bundled inputs.

**Keywords:** simulation calibration, test size, forecast comparison, cross-sectional dependence, nested models, Clark-West, data snooping, superior predictive ability, deflated Sharpe ratio, effective sample size, statistical power, transaction costs, reproducible research.

> **Review response.** [`docs/review_response.md`](docs/review_response.md) records an external review of the manuscript, the three claims it falsified, what was corrected, which objections did not hold, and what remains open. Three of its seven open items are now closed — simulation-calibrated inference, nested-model tests, and joint resampling of the search — and closing them retracted a published claim. The scale limitation is not closed and is the binding one: this is a prototype and a methods note, not a benchmark.

> **Manuscript.** [`paper.pdf`](paper.pdf) is the full-length preprint: the same results with the proofs, the appendices and the executed fold table. It is regenerated from the committed run bundle by `make paper`.

> **Scope.** This is not a historical BIST-100 study. The accepted experiment is named `fixed_bist_large_cap_prototype` and contains GARAN, ISCTR, KCHOL and THYAO. [`docs/component_status.yaml`](docs/component_status.yaml) is the machine-readable boundary between what is implemented, what is integrated, and what has empirical evidence behind it.

---

## Contents

[1. Introduction](#1-introduction) · [2. Related work](#2-related-work) · [3. Data](#3-data) · [4. Method](#4-method) · [5. Calibration](#5-calibrating-the-evaluation-stack) · [6. Results](#6-results) · [7. Discussion](#7-discussion) · [8. Limitations](#8-limitations) · [9. Conclusion](#9-conclusion) · [Reproducing](#reproducing) · [Repository layout](#repository-layout) · [References](#references)

---

## 1. Introduction

A short-horizon equity forecasting result is easy to manufacture and hard to verify. The standard failure modes are well catalogued: a target the backtest could not actually have traded, a validation split that lets a model see its own future, a metric whose null is never stated, a p-value computed on rows that are not independent, and a headline drawn from whichever configuration happened to look best. Each of these inflates apparent performance without leaving an obvious trace in the code.

There is a second failure mode, and it is the subject of this work. The catalogue above assumes the tests themselves work. A p-value is a claim about a long-run frequency, and that claim is a theorem only under conditions the data are supposed to satisfy: independent observations, non-nested models, trials that disperse like independent draws. Panel forecast comparison satisfies none of them. Four tickers observed on one session are not four observations. A fitted forecast compared against a zero forecast is nested inside it. Seventy-two configurations run over overlapping folds are not seventy-two searches. Each violation is well known in isolation; what is not known is how far the resulting numbers are from the numbers they claim to be, on data with the dependence structure these studies actually have.

Section 5 measures it, by generating panels whose truth is fixed by construction and running the production estimators on them unmodified. The answers are not marginal:

| Test, as usually applied | Nominal size | Measured size |
| --- | --- | --- |
| Diebold-Mariano on panel rows, 4 correlated names | 0.05 | **0.2284** |
| Diebold-Mariano on panel rows, 100 correlated names | 0.05 | **0.7913** |
| Holm across a family of six, row level, 30 names | 0.05 | **0.5875** |
| Diebold-Mariano vs a zero benchmark, fitted noise forecast | 0.05 | **0.9219** |
| False Strategy Theorem threshold, any trial correlation | — | **cleared ~half the time** |
| *Session aggregation + Clark-West + joint bootstrap* | 0.05 | **0.0385 – 0.0565** |

This repository is built the other way around from the usual study. The question is not *can a model be found that beats the benchmark*, but *if such a model existed, would this apparatus detect it, would it say so honestly, and does it*. Six commitments follow.

**The target is executable.** The label is the return the simulated portfolio can actually earn under its own stated timing: signal after the close of session $t$, execution at the observed open of session $t+1$, exit at that session's official close. Nothing is predicted that could not have been traded.

**The evaluation respects the panel's structure.** Folds are expanding windows over unique trading dates, never row offsets, with purge and embargo. All tickers on one date stay in the same partition. Preprocessing is fitted on training rows only.

**The sample size is measured rather than assumed.** Four tickers observed on the same session are not four independent observations. Section 6.1 quantifies this and Section 4.4 states what is done about it.

**Every test states its measured size beside its nominal size.** An estimator is not trusted because it is standard. Section 5 runs each one on data whose answer is known before the estimator sees it, and reports the rate at which it is wrong with a Monte Carlo interval.

**The search is counted.** Every arbitrary choice in the design — how many training dates, how wide a validation window, how large an embargo, how many names to hold — is swept, and the reported configuration is placed inside the resulting distribution rather than presented as *the* result.

**The design is asked what it could have found.** A negative result carries information only in proportion to the power of the design that produced it. "We tested and found no effect" and "we ran a test that could not have found the effect even had it been there" are different claims, and this literature almost never distinguishes them. Section 4.8 derives three bounds — on detectable accuracy, on what breadth can buy, and on what the cost schedule demands — and Sections 6.6 to 6.8 evaluate them on the committed run.

The empirical outcome is negative. It is retained because the calibration is what makes it interpretable, and because the same calibration shows that the uncorrected apparatus would have reported a *different* negative result — one a reader could reasonably have mistaken for a finding. A negative result delivered with the machinery that would have found a positive one is a stronger statement than a positive result delivered without it; a negative result that also reports what its own instruments do when the answer is known is stronger still.

---

## 2. Related work

**Forecast comparison.** Diebold and Mariano [1] give the standard test of equal predictive accuracy between two forecasts; Harvey, Leybourne and Newbold [2] show that the statistic is oversized in small samples and supply the correction used here. Diebold [3] later cautioned that the test compares *forecasts*, not the models that produced them, which is the interpretation adopted below.

**Data snooping.** White [4] shows that selecting the best of $m$ models and testing that model alone is invalid, and gives a bootstrap Reality Check for the joint null. Hansen [5] studentizes the statistic and recentres the bootstrap, gaining power and robustness to irrelevant alternatives; Romano and Wolf [34] extend the idea to a stepwise procedure that identifies which models survive, and Hansen, Lunde and Nason [35] to a confidence set over models. Sullivan, Timmermann and White [6] apply the Reality Check to a century of technical trading rules and find that the apparent performance of the best rule largely disappears once the size of the search is accounted for. That result is the direct template for Section 6.3.

**Nested models.** When one forecast nests another, the population loss differential is zero under the null and the Diebold-Mariano numerator degenerates; Clark and McCracken [31] and McCracken [32] derive the resulting non-standard limiting distributions. Clark and West [30] give the adjustment used here, which restores approximate normality by removing the term the fitted model's own estimation variance contributes. Giacomini and White [33] instead change the null, reframing the comparison as one of conditional predictive ability. This literature is mature and unambiguous, and it is nonetheless routine for applied short-horizon studies to compare a fitted model against a zero or historical-mean forecast with an unadjusted squared-error test. Section 5.3 reports what that costs.

**Cross-sectional dependence in panels.** Petersen [36] shows that standard errors in finance panels are badly underst

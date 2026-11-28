# Reporting checklist for GRU / sentiment stock-forecasting papers

Purpose: a reviewer or replicator should be able to answer every item below
from the manuscript alone. Each item states what to check, why it changes the
conclusion, and - where we tested it - what our own run showed.

Status key: **[must]** blocks a directional or profitability claim;
**[should]** blocks a comparison across papers; **[nice]** improves
reproducibility.

---

## A. Data and alignment

**A1 [must] State the price series and adjustment.** Adjusted close vs raw
close changes both returns and R^2. Say which, and the vendor.
*Our run:* Yahoo Finance, `auto_adjust=True`.

**A2 [must] Prove sentiment is lagged.** News after the close of day *t* must
not enter the features of day *t*. Publish the merge rule (e.g. `shift(1)` on
the tone series) and the timestamp convention.
*Our run:* GDELT tone for *t-1* used at *t*; inner join on trading dates.

**A3 [must] Report the intersection of price and text coverage.** Sentiment
coverage is usually shorter than price coverage; the evaluated period silently
shrinks. State both windows.
*Our run:* prices 2019-11 -> 2026-11, sentiment 2021-11 -> 2026-11; the shorter
window preserved one "significant" cell that the longer window reverses
(see E3).

**A4 [should] Report missing-data handling** and whether it uses future
information (forward-fill vs interpolation).

## B. Splits and scaling

**B1 [must] Chronological splits only**, with the split dates printed, not just
percentages.

**B2 [must] Scaler fit on training rows only**, refit at every walk-forward
origin. Fitting on the full sample is the most common silent leak.
*Field rate:* 3 of 1,423 audited abstracts mention it; 2 verified after reading.

**B3 [must] No target leakage through windows.** For an *h*-day target, the last
training window must end at least *h* days before the evaluation origin.

**B4 [should] Report whether hyper-parameters were tuned on the evaluation
period.** If yes, the reported score is optimistic.

## C. Targets and horizons

**C1 [must] Name the target and its horizon.** "Price" vs "return" vs "sign" are
different tasks; a price-level R^2 of 0.95 is compatible with chance direction.

**C2 [must] Match training target to reported horizon.** Training on a 1-day
return and reporting an *h*-day score is a mismatch, not a multi-horizon model.

**C3 [should] Report the target's base rate** (share of up moves) for the
evaluation sample. Without it, accuracy is uninterpretable.

## D. Evaluation windows

**D1 [must] State the window stride.** Overlapping *h*-day targets share
returns; the nominal *n* is then *h* times too large.
*Field rate:* 3 of 1,423 abstracts, 2 verified.

**D2 [must] Report *n* per cell** and the number of position changes.

**D3 [should] Report results per market and per horizon**, not only pooled.

## E. Metrics and inference

**E1 [must] Compare direction against the majority-sign base rate**, not 50 %.
*Our run:* base rates 0.60 / 0.64 / 0.54 at h = 1 / 3 / 5; GRU's 0.446 mean
accuracy is far below them.

**E2 [must] Use at least persistence and historical-mean benchmarks.** A GRU
that loses to the historical mean has no predictive content to explain.
*Our run:* 2 of 84 cells beat the mean after correction - both negatively
significant.

**E3 [must] Report window sensitivity.** Show the same specification over at
least two evaluation windows or sub-periods.
*Our run:* AAPL h=1 GRU = **+0.106 (Holm 0.003)** on 2024-2026 and **-0.204
(Holm 0.030)** on 2023-2026. One walk-forward number is a window, not a model.

**E4 [must] Correct for multiple testing** across the cells you report (Holm or
BH), and say which family was corrected.

**E5 [must] Use a serial-correlation-robust test** (DM with HAC / Newey-West),
not a *t*-test on daily errors.

**E6 [should] Average >= 3 seeds** and report the seed spread. A single-seed
difference smaller than the spread is noise.
*Our run:* mean GRU seed spread 0.032 - larger than most reported "gains".

**E7 [should] For paired claims (with vs without sentiment), use a paired test**
(DM on paired errors, McNemar on direction calls), not two independent scores.

## F. Economics

**F1 [must] Charge transaction costs** and state the assumption.
*Our run:* 5 bps/side, charged per position change.

**F2 [must] Report turnover.** High accuracy with daily flips can lose money.
*Our run:* Ridge 0.94 vs GRU 0.46 position changes per window.

**F3 [should] Separate drift from skill.** In a rising sample, long exposure
earns the mean return; compare with buy-and-hold and with the historical mean.

**F4 [should] Report a capacity / liquidity note** if claiming implementability.

## G. Sentiment-specific

**G1 [must] Report the unconditional association** between the sentiment signal
and the future return (correlation / IC), with an overlap-aware caveat.
*Our run:* Pearson |r| <= 0.043 (p >= 0.143), Spearman |rho| <= 0.048 (p >= 0.097) - no unconditional content.

**G2 [must] Ablate sentiment against an identical price-only model**, same
origins, same seeds, same target.

**G3 [should] Report the sentiment effect per market.** Pooling can hide a sign
flip.
*Our run:* Korea +0.14 to +0.16 (Holm 0.08-0.15), US -0.05 to -0.09
(Holm 0.013-0.044).

**G4 [should] Describe the text pipeline** (source, language, lexicon/model,
aggregation window). "Sentiment" is not a specification.

## H. Reproducibility

**H1 [should] Publish code, seeds, library versions and the exact commands.**

**H2 [should] Publish per-cell outputs** (not only aggregate tables), so the
correction can be recomputed.

**H3 [nice] State runtime and hardware** for the reported runs.

---

## Score sheet

Count the **[must]** items satisfied: A1-A3 (3), B1-B3 (3), C1-C2 (2),
D1-D2 (2), E1-E5 (5), F1-F2 (2), G1-G2 (2) = **19 must-items**.

| Must-items satisfied | Reading |
|---|---|
| 19/19 | Claims are auditable; directional/profitability statements are testable |
| 13-18 | Fix the missing items before claiming skill |
| < 13 | Treat reported accuracy as descriptive only |

Our own paper: the self-audit of 2026-11-28 (item-by-item against this
checklist, by a second reader) scored the four main deliverables:

| Deliverable | Before remediation | After remediation |
|---|---|---|
| RESULTS_PACK.md | 13/19 | 19/19 |
| decisive_multiday_findings.md | 13/19 | 19/19 (A2 merge-rule sentence added in the second pass) |
| PIF_Q1_submission.md | 12/19 (strict) | 17/19 + the named gaps closed in the second pass (G1 caveat, D2 table pointer, E7 paired test, G4 aggregation window, H1/H3) |
| Data_Workflow_Novelty.pdf | 16/19 | 19/19 |

The shared [must] gap was B3 (purge rule), now stated with its code reference
in all four documents, together with the data-source, coverage, merge-rule,
stride, seed, cost, turnover and correction-family statements each document
was missing. The PIF total still needs a final re-score before it is quoted:
the second-pass edits are not yet re-checked item by item, and A4
(missing-data handling) is open in every markdown deliverable. So "19/19" is
not claimed here. The checklist exists so the next paper can be checked in
minutes rather than replicated in months.

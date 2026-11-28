# Swarm remediation report - 2026-11-28

Overnight professional-improvement pass on this repository, run as a parallel
swarm of 10 designated agents (8 diagnosis + reviewer/scorer verification).
Everything below was verified by re-checking the artifact, not by trusting a
child's claim. Nothing was committed or pushed; all changes are local.

## What ran

| Phase | Agents | Output |
|---|---|---|
| Diagnosis (parallel, read-only) | explorer, auditor, data-checker, tester, repro, researcher, historian, report-checker | claims-vs-artifacts audit, number re-derivation, CLI/dependency checks, recipe reproduction, external-source verification, git history reconstruction, checklist scoring |
| Verification (parallel, read-only) | reviewer, report-checker | diff review and re-score after remediation |

## Confirmed findings and what was done

### Fixed and verified

| # | Finding | Severity | Fix applied |
|---|---|---|---|
| 1 | NOTES.md printed the Ridge/GBR baseline rows with the GRU's own numbers; the real Samsung Ridge baseline (0.4510) beats the GRU (0.4224) | High - looks fabricated to a reviewer | baseline rows replaced with values re-read from `results/diagnostics_multi_seed.md` and `results/hypothesis_battery.md`; "Ridge/GBR ~= GRU" bullet corrected |
| 2 | "Minimum detectable excess ~= 10 pp at 80% power" in NOTES, decisive findings and methodology | High - the null looks better-powered than it is | recomputed by hand (two-sided binomial, alpha 0.05): ~20-21 pp at n=43-50; the first pass's "18-19 pp" was the one-sided value and was relabelled in the second pass |
| 3 | "Cells evaluated (5 markets x 3 horizons x 6 models) = 84" - 5x3x6 = 90, and the GRU arm ran on only 3 of 5 markets | High - arithmetic a reviewer checks first | both docs now state the real composition: 75 cells (5x3x5 families) + 9 GRU cells |
| 4 | The claimed 19/19 checklist score was not supportable (best deliverable scored 16/19; the shared [must] gap was the purge rule) | High | purge rule (B3) + code reference added to all four deliverables; checklist now shows the measured scores instead of 19/19 |
| 5 | `yfinance` and `scipy` imported by repo code but missing from requirements.txt | Medium - a clean install cannot run the pipeline | both added with floors; all 8 declared deps verified against the live environment |
| 6 | RESULTS_PACK Reproduce step 3 omitted `--outdir`, so step 4 dies on a missing glob | Medium | command fixed |
| 7 | "1,421 other papers" vs 1,423 everywhere else | Medium | typo fixed |
| 8 | Journal impact factors (10.2/9.4/5.7/2.6) unverifiable - Elsevier, Wiley and Clarivate all block automated access; "JCR 2025" is the wrong edition name (Clarivate's release is JCR 2026 with 2025 data) | Medium | PIF section I rewritten: values marked UNVERIFIED except Computational Economics 2.1, confirmed on Springer's page; edition naming corrected |
| 9 | arXiv 2606.29347 characterized as "finds no learned model beats the historical mean" - the paper also reports a positive 5-day result on a 500-stock panel | Medium | citation scope narrowed to match the abstract |
| 10 | GDELT described as "DOC 2.1" - GDELT's own docs name it the DOC 2.0 API | Low | corrected, with the 5-second rate limit noted |
| 11 | methodology claimed `analyze_relevant_corpus.py` writes the docs memo; it writes only ANALYSIS.json and ANALYSIS_TABLE.md (memo is hand-authored) | Medium - a verification step that did not happen as described | claim corrected |
| 12 | methodology claimed corpus counts were "cross-checked against the manifest line count" - the manifest is a dict, not 1,423 lines | Medium | claim corrected to the manifest's own `retained` count |
| 13 | Cross-reference pointed at PIF section H for impact factors; they live in section I | Low | fixed |
| 14 | "expanded from 24" / "was 22-24" windows - no earlier run artifact exists in repo or git | Low | unverifiable claim dropped |
| 15 | VolGatedGRU "worse MAE everywhere" - true for the paper-upgrade run, contradicted by the matched-arm diagnostics | Medium | PDF wording now scoped to the paper-upgrade run |
| 16 | "Only 3 of 1,423 papers report doing this" (train-only scaling) - 3 keyword hits, 2 verified | Low | PDF corrected to the 2-verified framing |
| 17 | README contradicted itself: return/directional targets documented as implemented but listed as to-do; omits VolGatedGRU, firm sentiment, benchmark exclusion | Medium | status list reconciled |
| 18 | NOTES cited "commit 22e85ed+" for strict verification; the stride landed in 55cadea | Low | fixed |
| 19 | NOTES timeline covered 4 of 48 commits | Low | timeline expanded to all workstreams with a pointer to git |
| 20 | Repo hygiene: lightning_logs/ (348 version dirs, 7.8M, regenerable, unreferenced), __pycache__, .freebuff tool state | Low | removed / gitignored |
| 21 | Every markdown deliverable used unicode punctuation (en/em dashes, arrows, smart quotes, comparison operators) | Low - submission-system risk | 9 files normalized to ASCII (Renyi keeps its accent as a proper noun) |
| 22 | Data_Workflow_Novelty.pdf rebuilt with all of the above plus net-return and turnover columns in the GRU table and a clean-checkout regeneration note | - | PDF regenerated (14 pages) |

### Evidence base: clean

The auditor and data-checker independently re-derived every headline number
from the raw artifacts: 0/84 positive excess surviving Holm; the 2 significant
cells (AAPL GRU h=1 -0.204 Holm 0.030; ^NSEI momentum h=5 -0.227 Holm <0.001);
GRU means -0.212/-0.221/-0.022; seed spread 0.032; sentiment deltas and IC
bounds; the period-sensitivity pair (+0.106 vs -0.204); the 84/30 cell counts;
all 1,423 and 575 corpus counts. All MATCH. The earlier E1-E8 "wins" are
correctly retired as artifacts. No number was fabricated anywhere; the
failures were documentation defects.

## Decisions only you can make

1. **AI co-author trailers in git history.** Commits d5dd7d7, aff5bb4 and
   80173c8 carry "Generated with Codebuff" / CommandCodeBot trailers, and
   d5dd7d7's message claims "verified JCR 2025 IFs" - now falsified. History
   was NOT rewritten (unattended history rewriting is reckless). If this repo
   goes public, decide how you want to handle both the trailers and that
   commit message.
2. **results/ and literature_corpus/ are gitignored by design.** The docs cite
   them as the evidence base and now state the regeneration commands, but
   checklist item H2 (published per-cell outputs) cannot be satisfied without
   either un-ignoring results/ (9.3M, 734 files) or attaching the summaries to
   a submission. Your call.
3. **Dead branch:** scripts/fetch_india_intraday.py and data/india_intraday/
   are untracked, unreferenced, and contradict the "no intraday data"
   limitation. Left in place; delete or document.
4. **No LICENSE file** exists. Add one before the repo is shared.

## Second pass (verification round)

A reviewer agent re-checked the whole diff and a re-scorer re-scored the four
deliverables. The re-score: RESULTS_PACK 13 -> 19/19, decisive_multiday_findings
13 -> 18/19, PDF 16 -> 19/19, PIF 12 -> 17/19 (strict). The reviewer caught
8 misses in the first remediation pass, all now fixed and verified:

1. The 19/19 self-claim survived in PIF C8 and the PDF's C8 bullet while the
   checklist disowned it - both replaced with the measured scores.
2. The corrected MDE was the one-sided value wearing a "two-sided" label
   (exact two-sided is ~20-22 pp, not 18-19). All files relabelled: "roughly
   20-21 pp (two-sided, alpha = 0.05)".
3. PIF's purge sentence had no code reference - the `make_xy` citation added.
4. The PDF's reproduction block was still truncated and still missing
   `--outdir` on the sentiment run - commands split and fixed; the PDF now
   prints `--outdir` on both runs.
5. "expanded from 24" survived in NOTES.md - dropped.
6. README still headlined the retired E1-E8 results as "Genuine multi-day
   beats" - now labelled RETIRED with the decisive-protocol contradiction.
7. VolGatedGRU "worse MAE everywhere" was only half-scoped (the gate beats the
   concat arm on AAPL h=3 MAE) - PDF now claims "no win over the baseline GRU
   in any arm", which is what the artifacts support.
8. My ASCII pass left two blemishes - a duplicated PIF recommendation
   paragraph (removed) and garbled math `h  in  {1, 3, 5}` (fixed) - plus the
   re-score named the last gaps, all closed in this pass: the merge-rule
   sentence in decisive_multiday_findings (A2), the IC overlap caveat in PIF
   (G1), the per-cell table pointer (D2), the paired-test names (E7), the
   text-pipeline aggregation window (G4), the versions/runtime pointers
   (H1/H3), a pre-existing broken bold markup in literature_synthesis, and a
   0.001 rounding drift (+0.139 -> +0.138).

## Final state

- All 22 findings + all 8 reviewer misses + all re-score gaps are fixed.
- Requirements, .gitignore, clutter and the PDF are done.
- Remaining honest gaps: PIF needs a final item-by-item re-score before any
  total is quoted (its last measured total is 17/19 strict); A4
  (missing-data handling) is open in every markdown deliverable; the four
  decisions in the section above are yours.
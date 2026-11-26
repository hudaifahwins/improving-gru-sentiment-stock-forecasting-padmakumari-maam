# Method-driven literature expansion — 1,423-paper gap audit (2026-11-26)

Companion to `docs/literature_synthesis.md` (575-paper citation-lineage corpus).
This expansion was fetched with `scripts/fetch_relevant_papers.py` and audited
with `scripts/analyze_relevant_corpus.py`; every count below is computed from the
manifest's verbatim OpenAlex titles + abstracts, and no bibliographic field is
invented. Abstract-level only — a keyword hit is evidence of *mention*, not of
verified practice.

## 1. What was added

- Retained: **1,423** papers, **all with abstracts**, **993 open access (69.8%)**.
- Candidate records 1,474 · off-domain guard 51 · errors 0 · duplicates 0.
- Years: 2015–2026 (one record dated 2027), heavily recent (2025: 362, 2026: 431).
- Themes by query plan: sentiment 632 · targets_and_evaluation 220 ·
  volatility_and_regime 204 · forecasting_models 183 · markets 108 ·
  portfolio_and_microstructure 76.

Unlike the lineage corpus (2018–2026, co-citation with Padmakumari et al.), this
corpus is **method-driven**: queries target the eight gaps in §5 of the synthesis
(returns/directional targets, evaluation rigor, sentiment sources, Korea coverage,
volatility/volume inputs, attention/TCN baselines).

## 2. Gap audit

Counts over the 1,423 abstracts; the lineage column repeats the 575-paper figures
from `docs/literature_synthesis.md` where a comparable bucket exists. Keyword
sets differ slightly between the two audits, so deltas are indicative.

| signal | papers | share | lineage 575 | delta (pp) |
|---|---|---|---|---|
| `target_price_level` | 789 | 55.4% | 278 (48.3%) | +7.1 |
| `target_return` | 312 | 21.9% | 82 (14.3%) | +7.7 |
| `target_direction` | 297 | 20.9% | — | — |
| `target_volatility` | 124 | 8.7% | 17 (3.0%) | +5.8 |
| `input_volume` | 164 | 11.5% | 12 (2.1%) | +9.4 |
| `walk_forward` | 84 | 5.9% | — | — |
| `non_overlapping` | 3 (2 verified) | 0.2% | 0 (0.0%) | +0.2 |
| `significance_test` | 128 | 9.0% | 20 (3.5%) | +5.5 |
| `multiple_testing` | 18 | 1.3% | — | — |
| `train_only_scaling` | 3 (2 verified) | 0.2% | 0 (0.0%) | +0.2 |
| `transaction_costs` | 91 | 6.4% | — | — |
| `backtest` | 191 | 13.4% | — | — |
| `leakage` | 44 | 3.1% | — | — |
| `sent_finbert` | 138 | 9.7% | 53 (9.2%) | +0.5 |
| `sent_llm` | 92 | 6.5% | — | — |
| `sent_lexicon` | 43 | 3.0% | — | — |
| `sent_gdelt` | 10 | 0.7% | 0 (0.0%) | +0.7 |
| `sent_social` | 196 | 13.8% | — | — |
| `sent_news` | 282 | 19.8% | — | — |
| `sent_filings` | 21 | 1.5% | — | — |
| `market_korea` | 25 | 1.8% | 2 (0.3%) | +1.4 |
| `market_us` | 229 | 16.1% | — | — |
| `market_china` | 121 | 8.5% | — | — |
| `market_india` | 112 | 7.9% | — | — |
| `arch_gru` | 383 | 26.9% | 75 (13.0%) | +13.9 |
| `arch_lstm` | 674 | 47.4% | 254 (44.2%) | +3.2 |
| `arch_transformer` | 287 | 20.2% | 63 (11.0%) | +9.2 |
| `arch_gnn` | 78 | 5.5% | 35 (6.1%) | -0.6 |
| `arch_ensemble` | 331 | 23.3% | — | — |
| `arch_rl` | 56 | 3.9% | — | — |
| `baseline_persistence` | 44 | 3.1% | — | — |

Joint mentions (all 1,423): significance + walk-forward **24**;
significance + walk-forward + transaction costs **6**; walk-forward + costs
**13**; non-overlapping + significance **2**.

## 3. What the expansion changes — and what it does not

**Does change (grounded):**

1. **Return targets are no longer a fringe.** 312/1,423 (21.9%) mention return
   prediction vs 82/575 (14.3%) in the lineage corpus, while price-level targets
   still lead at 789 (55.4%). Our E1 return/directional work sits in a real but
   still-minority literature.
2. **GRU is well represented when you query for it.** 383/1,423 (26.9%) vs
   75/575 (13.0%) — a direct consequence of the GRU-targeted query plan, not a
   claim that the field shifted. LSTM remains the plurality architecture (47.4%),
   transformer 20.2%, GNN 5.5%.
3. **Sentiment sourcing is now visible.** FinBERT 138, LLM-based sentiment 92,
   social media 196, news 282, filings/earnings 21. **GDELT rises from 0/575 to
   10/1,423**, including FinBERT + GDELT headline pipelines; separately, the
   corpus now carries a Korean LLM-news-tone study. Our lagged GDELT tone
   ablation therefore has direct comparators rather than being unliteratured.
4. **Korea is no longer a two-paper niche.** 25/1,423 (1.8%) mention Korea/
   KOSPI/Samsung, and 8 titles are Korea-focused (e.g. LLM news tone and Korean
   returns, 2026; Korean ML return forecasting, 2023). Still small, but citable
   as a real sub-literature.
5. **Evaluation rigor is named more often.** Significance 128 (9.0%) vs 20
   (3.5%); walk-forward 84 (5.9%); transaction costs 91 (6.4%); leakage 44
   (3.1%), including a dedicated study of look-ahead bias in GPT-sentiment
   backtests.

**Does not change (the gaps survive):**

- **Non-overlapping multi-day evaluation is still essentially absent:** 3 keyword
  hits, **2 genuine** after manual reading (one hit is about non-overlapping
  *fiscal years*). Only the two 2026 papers below actually evaluate disjoint
  folds.
- **Train-only scaling is still essentially unreported:** 3 keyword hits,
  **2 genuine** (the third is a false positive where "MinMaxScaler" and
  "training" co-occur in a sentence that normalizes the whole dataset — the exact
  P1 pitfall).
- **Rigor combinations are rare:** only **6/1,423** abstracts mention
  significance testing, walk-forward, *and* transaction costs together; only
  **2** combine non-overlapping evaluation with significance testing.
- **Price-level targets still dominate** (55.4%), and direction classification
  (20.9%) is often reported without significance tests.

## 4. Closest works to our protocol (abstracts read, links from manifest)

1. **Statistical Accuracy Versus Trading Utility: A Multi-Horizon Evaluation of
   Generative-AI-Augmented Deep Learning on the KSE-100 Index** (2026, *IJIST*;
   DOI 10.33411/ijist/1918). Eleven configurations (Ridge, RF, XGBoost, LSTM, GRU,
   BiLSTM, PatchTST, β-VAE variants) under an identical expanding-window
   walk-forward, **13 non-overlapping test folds**, horizons 1/5/10/20, a
   cost-aware backtest (5 bps fees + 2 bps slippage), and Diebold–Mariano tests.
   The closest published analogue to our E2/E4 protocol.
2. **Does Regime-Gated Attention Help Stock Return Prediction? A
   Leakage-Controlled, Pre-Registered Evaluation** (2026, arXiv:2606.29347).
   Aligned and purged windows, identical test dates for all models, cost-aware
   non-overlapping backtests, five seeds, DM tests; **no learned model beats the
   historical-mean forecast**. Directly consonant with our h=1 equivalence result.
3. **Deep Learning Stock Index Forecasting Against Tuned Learners and Naive
   Baselines Under a Leakage-Free Protocol** (2026, *MAKE* 8(11):366;
   DOI 10.3390/make8110366). 3,910 out-of-sample sessions (2022–2025), SET100/
   S&P 500/CSI 300 as confirmatory markets, expanding windows, purged boundaries,
   **train-only scaling**, naive baselines, GRU/TCN/DLinear/PatchTST/MLP.
4. **The Illusion of Accuracy: Machine-Learning and GARCH Forecasts of UK Equity
   Index Prices** (2026, Research Square preprint; DOI 10.21203/rs.3.rs-11480555/v1).
   FTSE 100/250 weekly 2015–2026, expanding window, 251 out-of-sample weeks,
   random walk / historical mean / ARIMA / GARCH / predictive regression / RF / NN,
   evaluated on price errors plus formal return-predictability tests.
5. **Bio-Inspired Hyperparameter Optimization of LSTM Networks for Long-Horizon
   Stock Price Forecasting** (2026, *Forecasting* 8(5):84;
   DOI 10.3390/forecast8050084). Chronological splitting, **training-only scaler
   fitting**, validation-only hyperparameter selection, held-out test scored once.
6. **Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT
   Sentiment Analysis** (2023, arXiv:2309.17322). Shows LLM training-period /
   backtest overlap creates look-ahead bias *and* a distraction effect — the
   strongest argument yet for our causal-lag discipline when using LLM sentiment.
7. **Does News Tone from Large Language Models (LLMs) Predict Stock Returns?
   Evidence from Korea** (2026, *Korean Journal of Financial Studies*;
   DOI 10.26845/kjfs.2026.8.55.4.299). KrFinBERT/DistilBERT/RoBERTa/ELECTRA/
   Llama3 tones on Korean news predict next-day returns; long–short portfolios
   significant; BERT high–low portfolio 0.242%/day (5.324%/month). A direct
   Korea + LLM-sentiment comparator for our Samsung arm.
8. **Forecasting Korean Stock Returns with Machine Learning** (2023, *APJFS*;
   DOI 10.1111/ajfs.12419). Korean market 1987–2018 (32 years), median vs mean
   regression, GBM vs linear benchmarks — a long-horizon Korea baseline.
9. **Machine learning in the Chinese stock market** (2021, *JFE*;
   DOI 10.1016/j.jfineco.2021.08.017). Large factor set, liquidity as leading
   predictor, explicit transaction-cost analysis and out-of-sample economic
   performance.
10. **Integrating FinBERT based news sentiment and technical indicators for LSTM
    stock price forecasting** (2026, *JAEPS*;
    DOI 10.54254/2977-5701/2026.36910). 58,466 GDELT headlines for AAPL/AMZN/
    GOOG/MSFT/NVDA (2019–2023), temporally aligned company-level sentiment +
    technical indicators — an apples-to-apples GDELT comparator for E3.

## 5. Mapping to the prioritized experiments (E1–E8)

- **E1 (return targets):** supported and contextualized — returns 21.9% vs price
  level 55.4%; the KSE-100 and Korea LLM studies report return/trading-utility
  evaluation, not just price error.
- **E2 (non-overlapping multi-day):** gap confirmed — 2 verified papers out of
  1,423; both 2026, both cost-aware. Our disjoint-window reporting is still
  rare enough to be a contribution.
- **E3 (sentiment source ablation):** GDELT 10, FinBERT 138, LLM 92; the GPT
  look-ahead paper (2023) and the FinBERT+GDELT LSTM paper (2026) set the
  comparison bar for our lagged-tone ablation.
- **E4 (significance battery):** 9.0% mention significance, but only 6 combine it
  with walk-forward and costs; DM/Holm reporting remains differentiating.
- **E5 (cross-market):** Korea 25 mentions / 8 titles; China 121 and India 112
  are far denser, so the Korea arm is still the thinner replication.
- **E6 (leakage checklist):** 44 leakage mentions, including a dedicated GPT
  look-ahead study; purged/aligned windows and train-only scaling are the named
  controls to cite alongside our audit.
- **E7 (volume/volatility):** volume 11.5% vs 2.1% lineage, volatility 8.7% vs
  3.0% — both now have enough exemplars to justify the feature expansion.
- **E8 (attention/TCN baseline):** transformer 20.2%, GNN 5.5%, RL 3.9%; the
  regime-gated attention paper's negative result is the closest single
  comparator and argues for testing attention once under our protocol, as
  planned.

## 6. Limits

1. Abstract-level only; keyword presence is *mention*, not verified methodology.
2. The three rarest signals were read manually: `non_overlapping` 3 hits → 2
   genuine; `train_only_scaling` 3 hits → 2 genuine. All other counts are
   regex-only.
3. Retrieval is query-based and deliberately weighted toward GRU/sentiment/
   evaluation, so cross-corpus deltas reflect both the literature and the query
   plan; they are not field-wide trend estimates.
4. OpenAlex metadata as of 2026-11-26; one record carries a 2027 date. Citation
   counts are OpenAlex values at fetch time.
5. Full texts were not bulk-downloaded; `oa_url`/`oa_pdf_url` mark the accessible
   subset (993 papers).
6. The arXiv supplement run (`literature_corpus/arxiv_2026-11-26/`) produced no
   records and is not part of this audit.

## 7. Reproduction

```bash
python3 scripts/fetch_relevant_papers.py --target 1500 \
    --out literature_corpus/relevant_2026-11-26
python3 scripts/analyze_relevant_corpus.py \
    --corpus literature_corpus/relevant_2026-11-26
# -> ANALYSIS.json, ANALYSIS_TABLE.md (both gitignored with the corpus)
```

Counts in this memo match the 1,423-record run of 2026-11-26. Re-running the
fetch (OpenAlex index moves) can change the totals.

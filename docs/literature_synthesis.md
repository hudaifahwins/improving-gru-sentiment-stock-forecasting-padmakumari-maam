# Literature Corpus Synthesis — 575-Paper Corpus Anchored to Padmakumari et al. (2026)

**Generated:** 2026-11-23 · **Corpus:** `literature_corpus/` (575 papers, gitignored bulk) · **Manifest:** `literature_corpus/_manifest.json`
**Anchor paper:** Padmakumari P., Pranav R. Swaminathan, Harshavardhan S.R., Viswadarsini S., Jaiaakaash J.S. (2026). *Precision Investing: Combining Gated Recurrent Unit and Sentiment Analysis for Enhanced Stock Market Predictions.* Computational Economics. DOI: 10.1007/s10614-026-11393-w (OpenAlex `W7171375780`).
**Pipeline:** `scripts/fetch_literature_md.py` (v4) → `scripts/enrich_seed_abstracts.py` → `scripts/extract_corpus_findings.py` (12 batch notes in `literature_corpus/_batches/`) → `scripts/build_synthesis_pack.py`.

This document synthesizes the corpus **only from verified file contents** (frontmatter metadata + abstracts). No titles, authors, venues, DOIs, or experimental numbers are invented. Where a claim cannot be grounded, it is marked as a limit.

---

## 1. Corpus construction and integrity accounting

**Definition of relevance.** A paper belongs in this corpus if it is bibliographically tied to the faculty paper: (a) one of its 18 OpenAlex references (the paper's own bibliography), (b) one of OpenAlex's 10 related works, (c) a work that **cites** one of the 18 seeds (citing lineage, `filter=cites:`), or (d) a tight backfill match on `title_and_abstract.search` over 14 domain queries. This is a co-citation-lineage corpus, not a keyword scrape.

**Final composition (575 papers):**

| Source | Count | Share |
|---|---|---|
| Citing lineage (`cites:` of the 18 seeds) | 404 | 70.3% |
| Topic backfill (`title_and_abstract.search`, 14 queries) | 144 | 25.0% |
| Seed references of the faculty paper | 18 | 3.1% |
| OpenAlex related works of the faculty paper | 9 | 1.6% |

**Rejection accounting (208 records rejected before writing):** 38 duplicate OpenAlex IDs · 4 duplicate normalized titles (OpenAlex carries duplicate records) · 141 off-domain at fetch (finance-token guard) · 25 culled in a post-fetch audit (12 passed only via the verb "investigat*" — hate-speech, drought, sleep-wearable, museum-visitor studies; 13 had zero domain tokens in title — malicious-website detection, biopolymer process optimization, horticulture DL review, generic LSTM/TS surveys, ERP assistant, etc.). Culled files are preserved in `literature_corpus/_excluded/` for audit.

**Coverage:** 570/575 abstracts (99.1%) · 400/575 open access (69.6%) · years 2018–2026.
**Year histogram:** 2018: 6 · 2019: 5 · 2020: 8 · 2021: 31 · 2022: 46 · 2023: 66 · 2024: 163 · 2025: 178 · 2026: 72 (59.3% of the corpus is 2024–2025 — the corpus is current).
**Top venues:** IEEE Access (16), Mathematics (12), Electronics (9), Decision Analytics Journal (8), Procedia Computer Science (7), PeerJ Computer Science (6), PLoS ONE (6).

**Honesty notes.**
- 5 of the 18 seed references are paywalled Elsevier/Springer records with **no abstract in OpenAlex, Crossref, or Semantic Scholar** (S2 was rate-limited; publisher pages bot-walled): Rajendiran et al. 2021 (Materials Today Proceedings), Quadir et al. 2022 (Applied Soft Computing, cited 228), Hu et al. 2023 (Expert Systems with Applications), Lu et al. 2023 (Information Sciences, cited 246), Billah et al. 2024 (Neural Computing and Applications, cited 101). They are retained with full metadata (title/authors/year/venue/DOI/citations) and **excluded from abstract-level claims**. Two further corpus papers were enriched from publisher pages: Shah et al. 2024 (seed; Springer article page) and Thorat et al. 2023 (related work; Springer chapter page).
- Signal detection runs on title+abstract+venue+topics (not full text — mostly paywalled). Findings below are therefore claims about **what the literature reports in abstracts**, which is itself the unit of analysis for reporting-practice gaps.
- `cited_by` counts are OpenAlex values as of the fetch date.

---

## 2. Corpus at a glance (signal counts over 575 papers)

**Architectures / model families:** LSTM 254 (44.2%) · GRU 75 (13.0%) · CNN 69 (12.0%) · Transformer 63 (11.0%) · SVM/SVR 62 (10.8%) · Ensemble/stacking 60 (10.4%) · Vanilla RNN 60 (10.4%) · Random forest 58 (10.1%) · LSTM-GRU hybrid 53 (9.2%) · BERT/FinBERT 53 (9.2%) · Attention mechanism 52 (9.0%) · ARIMA/SARIMA 47 (8.2%) · BiLSTM 35 (6.1%) · Graph NN 35 (6.1%) · Ridge/linear 32 (5.6%) · XGBoost 27 (4.7%) · MLP 23 (4.0%) · Reinforcement learning 22 (3.8%) · CNN-LSTM/GRU 22 (3.8%) · kNN 19 (3.3%) · GARCH 17 (3.0%) · BiGRU 11 (1.9%) · TCN 11 (1.9%) · GAN 10 (1.7%) · Momentum/technical rule 8 (1.4%) · Prophet 7 · Autoencoder 7 · Persistence/naive 6 (1.0%) · LightGBM 5 · Modern time-series Transformer 5 · TFT 3 · VAE 1 · Echo state 1.

**Input modalities / feature engineering:** Twitter/social 103 (17.9%) · News articles 101 (17.6%) · Technical indicators 85 (14.8%) · OHLCV prices 82 (14.3%) · Fundamentals 60 (10.4%) · Indices 55 (9.6%) · Macro 38 (6.6%) · Crypto 23 (4.0%) · Forex 19 (3.3%) · Commodities 16 (2.8%) · Volume/order book 12 (2.1%) · Options/derivatives 8 (1.4%) · Realized/implied volatility 6 (1.0%) · Search trends 2 (0.3%) · **GDELT 0 (0%)**.

**Targets:** Price level 278 (48.3%) · Direction classification 123 (21.4%) · Returns 82 (14.3%) · Volatility 17 (3.0%).

**Horizons (explicitly stated in abstract):** Monthly/long-term 80 (13.9%) · Next-day h=1 30 (5.2%) · Intraday 17 (3.0%) · Multi-day h=3/5+ 9 (1.6%) · **no horizon keyword 452/575 (78.6%)**.

**Evaluation metrics:** Classification accuracy 318 (55.3%) · RMSE 97 (16.9%) · MAE 79 (13.7%) · MAPE 54 (9.4%) · Returns/Sharpe 39 (6.8%) · R² 34 (5.9%) · Statistical tests 20 (3.5%) · Directional accuracy 12 (2.1%) · Other 3.

**Good practices:** Baseline included 96 (16.7%) · Significance tested 21 (3.7%) · Walk-forward/OOS 9 (1.6%) · Train-only scaling 0 (0%).

**Gap/risk signals:** India 46 · US market 43 · China/A-share 41 · Single ticker/market 33 · Scaling/normalization mentioned 31 · Crypto market 17 · Emerging markets 8 · Look-ahead/leakage discussed 5 · Korea 2 (0.3%) · Overlapping windows 2 (both non-leakage contexts: ViT patch tiling in Friday et al. 2025; MODWT wavelet overlap in Ghosh et al. 2025).

---

## 3. Cross-corpus themes (grounded in the texts)

### T1. LSTM dominates; GRU is the efficient minority — the faculty paper sits in the GRU tradition
LSTM appears in 254 abstracts (44.2%) vs GRU in 75 (13.0%). The corpus's own evidence supports GRU's efficiency rationale: Thorat et al. (2023, LNNS — a related work of the faculty paper) — "BiLSTM performed better than GRU… however, GRU was almost twice as fast as the BiLSTM model… due to its simpler architecture"; **Saud et al. (2020, Procedia Computer Science, cited 135**, seed**)** — GRU "most successful" vs Vanilla RNN/LSTM on NEPSE banking stocks, with look-back period analysis; **Shahi et al. (2020, Mathematics, cited 258)** — normalized LSTM-vs-GRU comparison under identical conditions with significance assessment; **Farhadi et al. (2025, IEEE Access, cited 79)** — hybrid LSTM-GRU leveraging "the full range of technical analysis features"; **Vancsura et al. (2025, Forecasting, cited 74)** — PRISMA SLR identifying LSTM, GRU, XGBoost and hybrids as the dominant predictive models across equities, crypto, commodities and FX. Implication: a leak-free GRU audit is positioned in the corpus's efficiency-motivated minority, not the LSTM majority — and the corpus gives us named, citable precedents for GRU competitiveness.

### T2. Sentiment integration is real but shallow in time discipline; GDELT is corpus-absent
Social (103) and news (101) modalities are the #1/#2 text sources; BERT/FinBERT appears in 53 abstracts. Representative grounded works: **Memi et al. (2024, Applied Sciences**, seed**, cited 49)** — BIST100 financial tweets, CNN best at 83.02% binary; **Mehta et al. (2021, PeerJ CS, cited 226)** — harvesting social-media sentiment with deep learning; **Koukaras et al. (2022, Telecom, cited 127)** — Twitter+StockTwits on Microsoft; **Varghese et al. (2023, Heliyon**, seed**)** — modified VADER + Shannon/Rényi entropy causality between news sentiment and Indian pharma prices (COVID); **Zhang et al. (2023, ACM TKDD**, seed**, cited 44)** — DGATS dynamic graph attention with Kronecker tensor fusion of sentiment + transactions + text; **Das et al. (2024, Decision Analytics Journal**, seed**, cited 98)** — GNN+sentiment survey naming data collection/preprocessing/annotation as the open challenges; **Lee et al. (2024, Scientific Reports, cited 113)** — ESG sentiment + technical indicators; **Thota et al. (2025, ICICKE, cited 42)** — FinBERT + TFT; **Qian et al. (2025, PLoS ONE, cited 24)** — incremental-learning transformer for online prediction. **Zero papers mention GDELT** — our `src/sentiment_gdelt.py` ingestion (1-trading-day causal lag, date-aligned tone) has no corpus precedent to copy from; we are introducing the source, not optimizing it. Also note seed **Aakash et al. (2024, Procedia CS)** is a *product-review* sentiment paper (LSTM/GRU F1 90–91%) — the faculty bibliography itself borrows sentiment methods from outside finance, which is exactly the kind of cross-domain transfer our audit makes explicit.

### T3. Price-level targets dominate; return/directional targets are the minority practice
Price level 278 (48.3%) vs direction 123 (21.4%) vs returns 82 (14.3%) vs volatility 17 (3.0%). The corpus — like the faculty paper — mostly regresses price levels. Our return-target diagnostics (`--return-type log|pct|logvol|pctvol`, hybrid loss, dead-zone filtering) align with the minority practice and with the corpus's own stress evidence: **Shahi et al. (2020)** shows level-error metrics can be dominated by persistence; our verified harness shows the same (raw-price GRU AAPL RMSE 17.64 vs persistence 4.38; return-GRU matches persistence on levels while retaining direction signal).

### T4. Multi-day horizons are barely evaluated, and non-overlapping evaluation is essentially unreported
Only 9 abstracts state multi-day horizons; 30 state next-day; 78.6% state no horizon at all. **No corpus paper reports non-overlapping (disjoint) multi-day evaluation** — the 2 "overlap" hits are ViT patch tiling (Friday et al. 2025) and MODWT wavelets (Ghosh et al. 2025), not window overlap. The closest rigorous practice: **Friday et al. (2025, IEEE Access)** — ViT+TFT over candlestick images + HOG + TFT, time-series cross-validation, **8 indices** (BSE, IXIC, N225, NIFTY-50, NSE-30, NYSE, S&P 500, SSE), horizons 1/3/7/10 days, MCC 0.9367 (claims we treat as optimistic but methodologically aware). This is the strongest corpus evidence that our `--stride nonoverlap` + causal-momentum baseline + small-n disclosure fills a real reporting void: overlapping h=3/5 DirAcc (AAPL 0.7762/0.7839) vs disjoint (0.6494/0.7333) vs momentum (0.5086/0.5000) is a measurement the corpus does not currently publish.

### T5. Accuracy-chasing culture; significance testing and directional accuracy are rare
Classification accuracy appears in 318 abstracts (55.3%) — frequently direction classification reported as bare "accuracy" — while statistical tests appear in only 20 (3.5%) and directional accuracy in 12 (2.1%). Inflationary claims exist in the corpus itself: **Raju et al. (2024, IC2PCT**, seed**)** reports "85% accuracy" plus "10.28% return in 3 months and 175% over 1 year"; **Yenikar et al. (2023, ESCI**, seed**)** lists 13 models with accuracies 70–95.11% on news polarity. Against this, **Shahi et al. (2020)** is the corpus template for honest comparison (normalized conditions + significance), and **Peng et al. (2021, Machine Learning with Applications, cited 124)** shrinks a 124-indicator factor zoo with feature selection under walk-forward + baseline + significance. The faculty paper's headline accuracy-style claims sit in the 55.3% majority; our audit's nRMSE/DirAcc/dead-zone-ε/seed-std reporting is the corrective the corpus itself lacks.

### T6. Leakage controls are named by a handful of 2025–2026 works — and they validate our audit design
Five abstracts discuss look-ahead/leakage explicitly: **Sun et al. (2025, Scientific Reports, cited 32)** — R-CSAN decoder "masking mechanism to prevent future information leakage", cross-market (Amazon, Maotai, Ping An, Vanke), full ablation table; **Manhas (2026, Zenodo dissertation)** — "particular emphasis… on preventing information leakage through chronological data handling"; **Minocha et al. (2026, IJSHRE)** — FinBERT sentiment + decay weights + volatility/recent-return features, "rolling time-series validation… helps eliminate look-ahead bias", 1–5 day horizons, 62% combined vs 48% sentiment-only vs 52% market-only; **Niharmine et al. (2026, IRASET)** — Moroccan MASI, 45k French/Arabic articles, "strict temporal separation… news sentiment computed only from articles published before each trading day's market open", CNN-LSTM 78.6% direction, R² 0.847, SHAP; **Theodorakopoulos et al. (2026, Electronics)** — critical review: "performance can be inflated by attention confounds, temporal leakage, and visibility effects", demanding "alignment discipline", "time-aware evaluation and auditable pipelines", and reporting checklists. These five are the corpus's methodological conscience — and they independently converge on exactly the controls our harness enforces (train-only scaling, chronological splits, causal sentiment lag, alignment probes, single test-set scoring).

### T7. Market coverage is India/US/China-heavy; Korea is nearly absent
India 46 · US 43 · China 41 · single ticker/market 33 · emerging markets 8 · **Korea 2 (0.3%)**. The faculty paper's Samsung (005930.KS) study therefore sits in the corpus's thinnest market niche. Our dual-market design (AAPL + Samsung) with per-market reporting is corpus-rare, and the Korea gap is itself a citable motivation (corpus Korea mentions: 2/575).

### T8. The architecture frontier is attention/transformers/graphs — but the corpus warns that novelty ≠ validity
Transformer 63 · attention 52 · BERT/FinBERT 53 · Graph NN 35 · TCN 11 · GAN 10 · modern TS transformers 5 · TFT 3. Frontier exemplars: **Fan et al. (2024, ISCMI, cited 80)** xLSTM for long-term stock forecasting; **Patel et al. (2024, ACM Computing Surveys, cited 79)** GNN review; **Dong et al. (2024, KDD, cited 184)** FNSPID financial-news dataset; **Ramzan et al. (2024, AI, cited 88)** GAN synthetic financial data; **Chen et al. (2024, Neural Computing & Applications, cited 84)** deep fusion of headlines + time series; **Zhao et al. (2023, Mathematics, cited 102)** relational models; **Ayyldiz et al. (2024, Heliyon, cited 91)** ML comparison across NYSE100/NIKKEI225/FTSE100/CAC40/DAX30/FTSE MIB/TSX; **Giantsidi et al. (2025, Int. Review of Economics & Finance, cited 122)** review of 187 Scopus studies (2020–2024); **Bao et al. (2024, Information Fusion, cited 219)** data-driven neural stock forecasting review. Against this, Theodorakopoulos et al. (2026) explicitly finds gains "often depend more on alignment discipline and conservative attribution than on architectural novelty" — which is precisely our h=1 equivalence finding (GRU = Ridge = GBR = momentum at the daily noise floor: AAPL 0.4832 ± 0.0089, Samsung 0.4224 ± 0.0014).

---

## 4. Methodological pitfalls mapped to our audit and the faculty paper

| # | Pitfall | Corpus evidence (grounded) | Faculty paper (2026) | Our leak-free audit |
|---|---|---|---|---|
| P1 | Whole-dataset scaling | 31/575 mention scaling/normalization; **0/575 report train-only fitting** | MinMax on whole dataset before split (per our audit) | Scaler fit on train rows only; same fitted transform applied downstream |
| P2 | Overlapping multi-day evaluation | 9 abstracts mention multi-day; **0 report non-overlapping evaluation** | Multi-day evaluation not reported — added by our audit | `--stride nonoverlap`; disjoint windows; small-n disclosed (~tens of points); causal momentum baseline |
| P3 | Accuracy inflation / R²-as-accuracy | 318/575 report classification accuracy; only 20/575 significance tests; Raju 2024 "85% + 175% return" | Reports MAPE/nRMSE/R²; no directional or significance metrics (audit matrix) | DirAcc with dead-zone ε, strict(0.005) variant, chance line 0.5, seed std (42/43/44), 48/48 runs |
| P4 | No persistence/momentum baseline | Persistence/naive in 6 abstracts; momentum rules in 8 | No persistence baseline per audit | Persistence + causal momentum baselines (AAPL h=1 momentum 0.4841; Samsung 0.4837) |
| P5 | Sentiment alignment undisciplined | Theodorakopoulos 2026 (temporal leakage, alignment discipline); Niharmine 2026 (pre-open news only); Minocha 2026 (decay weights, rolling validation) | Categorical sentiment fusion rule | 1-trading-day causal lag; date-aligned GDELT tone; standalone Δ reporting (AAPL −0.0022 ± 0.0154; Samsung 0.0000 ± 0.0000); zero-feature dummy control kept separate |
| P6 | Unreported hyperparameters | Most abstracts silent on architecture config | GRU hidden size/layers unreported | hidden_size=50, num_layers=1 disclosed; CLI reproducible |
| P7 | Single-market overfitting | Single ticker/market 33; Korea 2/575 | Samsung-focused | Dual market AAPL + Samsung, per-market tables |
| P8 | Volume/volatility inputs underused | Volume/order book 12; realized volatility 6 | Close-only features | Known gap: our loader drops OHLCV columns → experiment E7 |

---

## 5. Prioritized next experiments (each grounded in corpus evidence)

- **E1 — Return/directional targets with volatility-aware evaluation** (corpus: price-level 48.3% vs returns 14.3% vs volatility 3.0%; Shahi 2020 persistence-dominance evidence). Run `--loss hybrid --return-type logvol` at h=1 with strict ε; report dead-zone DirAcc vs persistence. *Already partially run: hybrid 0.4957 ± 0.0209 AAPL; logvol 0.4822 ± 0.0049.*
- **E2 — Non-overlapping multi-day evaluation as a reporting standard** (corpus: 0/575). Extend `--stride nonoverlap` to h=3/5 with momentum + persistence baselines; publish disjoint n per fold. *Verified so far: AAPL h=3 0.6494 ± 0.0347 / h=5 0.7333 ± 0.0526 vs momentum 0.5086/0.5000.*
- **E3 — Domain sentiment vs GDELT tone ablation** (corpus: BERT/FinBERT 53, GDELT 0, news 101; FNSPID dataset available — Dong et al. 2024 KDD, cited 184, in corpus). Ingest a labelled financial-news sentiment source (FNSPID-style), ablate against lagged GDELT tone, report dead-zone DirAcc. Extends our verified ablation: +GDELT tone AAPL 0.5067 ± 0.0303 (within seed noise), standalone Δ −0.0022 ± 0.0154.
- **E4 — Significance testing battery** (corpus: 3.5%). Diebold–Mariano / Wilcoxon across seeds 42/43/44 for GRU vs Ridge vs momentum; template = Shahi et al. 2020 (normalized comparison + significance) and Peng et al. 2021.
- **E5 — Cross-market replication** (corpus: Korea 0.3%; Friday et al. 2025's 8-index TSCV). Add a second US ticker + keep Samsung; per-market tables; time-series cross-validation per Friday/Manhas/Minocha.
- **E6 — Leakage reporting checklist** (corpus: Theodorakopoulos 2026 checklists; R-CSAN masking; Niharmine pre-open rule). Adopt as standard repo reporting: split chronology, scaler-fit location, sentiment timestamp alignment (our maxdiff = 0.0 probe), window stride, single test scoring.
- **E7 — Volume + realized-volatility inputs** (corpus: volume 12, realized vol 6; our loader drops OHLCV — known gap). Add volume features and realized-volatility inputs to `src/features_research.py`; ablate on both tickers.
- **E8 — One attention/TCN baseline on identical splits** (corpus: attention 52, TCN 11; R-CSAN ablations show temporal-module removal costs +38.6% RMSE). Hypothesis from our equivalence result: attention helps multi-day disjoint evaluation, not h=1. Test once, report honestly.

---

## 6. Exemplar papers (all present in `literature_corpus/`; metadata verified from manifest)

**Faculty paper's own bibliography (seeds):** Vijh et al. 2020 Procedia CS (cited 841, ANN+RF, RMSE/MAPE) · Saud et al. 2020 Procedia CS (135, RNN/LSTM/GRU look-back, NEPSE) · Akhtar et al. 2022 J. King Saud Univ. Science (137, RF+SVM, 80.3% accuracy claim) · Quadir et al. 2022 Applied Soft Computing (228, multi-layered sequential LSTM + optimization) · Lu et al. 2023 Information Sciences (246, TRNN) · Zhang et al. 2023 ACM TKDD (44, DGATS graph attention + Kronecker fusion) · Varghese et al. 2023 Heliyon (26, VADER + entropy causality) · Yenikar et al. 2023 ESCI (8, 13-model news-polarity comparison) · Sang et al. 2024 Mathematics (109, AMV-LSTM attention variant) · Billah et al. 2024 Neural Computing & Applications (101, moving-average comparison) · Das et al. 2024 Decision Analytics Journal (98, GNN+sentiment survey) · Memi et al. 2024 Applied Sciences (49, BIST100 tweets, CNN 83.02%) · Qin et al. 2021 ICBAIE (35, ARIMA+BP, JD/PDD) · Rajendiran et al. 2021 Materials Today Proceedings (27, survival study) · Aakash et al. 2024 Procedia CS (27, URL-based product-review sentiment, LSTM/GRU F1 90–91%) · Hu et al. 2023 ESWA (31, MSRL-Net aspect sentiment) · Shah et al. 2024 IJSAEM (17, social/web media sentiment review) · Raju et al. 2024 IC2PCT (9, SLR with inflated return claims).

**Lineage standouts:** Shahi et al. 2020 Mathematics (258, normalized LSTM/GRU + significance) · Mehta et al. 2021 PeerJ CS (226, social-media sentiment harvesting) · Bao et al. 2024 Information Fusion (219, neural stock forecasting review) · Dong et al. 2024 KDD (184, FNSPID news dataset) · Shaban et al. 2023 Neural Computing & Applications (129, SMP-DL) · Koukaras et al. 2022 Telecom (127, Twitter/StockTwits) · Peng et al. 2021 Machine Learning with Applications (124, 124-indicator factor zoo, walk-forward) · Giantsidi et al. 2025 Int. Rev. Economics & Finance (122, 187-study DL review) · Lee et al. 2024 Scientific Reports (113, ESG sentiment + technicals) · Chang et al. 2024 Electronics (104, DL+ML economic trends) · Peivandizadeh et al. 2024 IEEE Access (96, transductive LSTM + social sentiment) · Ramzan et al. 2024 AI (88, GAN synthetic finance data) · Bhuiyan et al. 2025 Array (86, algorithmic-trading DL SLR) · Chiong et al. 2022 IEEE TCSS (85, ensemble + sliding window + news) · Fan et al. 2024 ISCMI (80, xLSTM) · Patel et al. 2024 ACM Computing Surveys (79, GNN review) · Farhadi et al. 2025 IEEE Access (79, LSTM-GRU + full technical features) · Vancsura et al. 2025 Forecasting (74, PRISMA SLR, 4 asset classes) · Vuong et al. 2024 Science Progress (77, bibliometric review) · Millea 2021 Data (75, DRL trading survey) · Zhao et al. 2023 Mathematics (102, relational models) · Chen et al. 2024 Neural Computing & Applications (84, news+time-series deep fusion) · Ayyldiz et al. 2024 Heliyon (91, 7 developed-market indices) · Wu et al. 2021 ESWA (67, transfer learning + DL) · Srijiranon et al. 2022 Applied Sciences (62, PCA-EMD-LSTM + sentiment, Thailand) · Latif et al. 2024 Heliyon (53, PLSTM-TAL) · Wei et al. 2024 JTAER (55, dual-channel GAT fraud) · Thota et al. 2025 ICICKE (42, FinBERT+TFT) · Switrayana et al. 2025 JTIM (19, scaling-method comparison) · Qian et al. 2025 PLoS ONE (24, incremental transformer) · Xu et al. 2025 ICCMT (18, attention-augmented RNN) · Friday et al. 2025 IEEE Access (17, ViT+TFT, TSCV, 8 indices, h=1/3/7/10) · Sun et al. 2025 Scientific Reports (32, R-CSAN decoder masking) · Theodorakopoulos et al. 2026 Electronics (social-network + graph DL review naming temporal leakage) · Manhas 2026 Zenodo dissertation (chronological leakage prevention) · Minocha et al. 2026 IJSHRE (FinBERT + rolling validation, look-ahead elimination) · Niharmine et al. 2026 IRASET (Morocco MASI, strict temporal separation, pre-open news, SHAP).

---

## 7. Limits of this synthesis

1. Abstract-level analysis (full text mostly paywalled); reporting-practice findings are about abstracts.
2. 5 seed references lack abstracts everywhere we checked (listed in §1); their contributions are cited by title/venue/citations only.
3. Lineage-defined corpus: co-citation with the faculty bibliography is a strong but not exhaustive relevance definition; a PRISMA-style SLR would widen recall.
4. OpenAlex metadata (venues, citation counts) as of 2026-11-23; 25 post-audit culls preserved in `literature_corpus/_excluded/`.
5. Our verified metrics are quoted from the committed harness only: 48/48 runs; Samsung nRMSE 0.038 vs paper 0.048; h=1 DirAcc AAPL 0.4832 ± 0.0089, Samsung 0.4224 ± 0.0014; h=3/5 overlap vs `--stride nonoverlap` vs causal momentum; sentiment ablations as in §4 P5.

## 8. Reproduction

```bash
python3 scripts/fetch_literature_md.py --target 600 --out literature_corpus   # OpenAlex v4 (guards + dedupe)
python3 scripts/enrich_seed_abstracts.py                                      # Crossref → Semantic Scholar backfill
python3 scripts/extract_corpus_findings.py                                    # 12 batch notes + stats
python3 scripts/build_synthesis_pack.py                                       # stratified reading pack
```

# External 200+ Papers — Processed Analysis

- Source: OpenAlex (via `scripts/fetch_external_200plus.py`).
- Total records: **200** (years 2018–2026).
- With abstract: **200** (100.0%) · Open access URLs: **112** (56.0%).

## Exact keyword / regex counts across the 200 abstracts

| topic | records_with | pct_of_records | occurrences |
|---|---|---|---|
| gru | 25 | 12.5% | 48 |
| lstm | 119 | 59.5% | 234 |
| return_target | 31 | 15.5% | 72 |
| price_level | 135 | 67.5% | 326 |
| multi_day | 7 | 3.5% | 12 |
| leakage | 2 | 1.0% | 2 |
| significance | 38 | 19.0% | 47 |
| news_sentiment | 62 | 31.0% | 209 |

## Caveats / labeling

All findings above are **abstract-only** unless an OA full text was retrieved; dimensions not matched are `not-stated-in-abstract`.

## Notable papers (explicitly discussing multi-horizon, returns/leakage, or news/GDELT sentiment)

| openalex_id | title | year | matched_topic | is_oa |
|---|---|---|---|---|
| W2901852093 | Deep Learning for Stock Market Prediction Using Event Embedding and Technical Indicators | 2018 | return_target | False |
| W2890056567 | Stock Market Embedding and Prediction: A Deep Learning Method | 2018 | return_target | False |
| W2902982515 | Stock Return Prediction Using Artificial Neural Networks and Google Search Volumes | 2018 | return_target | True |
| W2922302024 | Designing an Optimal Portfolio for Iran's Stock Market with Genetic Algorithm using Neural Network Prediction of Risk and Return Stocks | 2019 | return_target | True |
| W3135862768 | Analysing Stock Market Trend Prediction using Machine & Deep Learning Models: A Comprehensive Review | 2020 | return_target | False |
| W3121535229 | Artificial Neural Networks in Stock Return Prediction: Testing Model Specification in a Global Context | 2020 | return_target | True |
| W3009443851 | Non-stationary neural network for stock return prediction. | 2020 | return_target | False |
| W3197302096 | Stock Market Prediction and Investment using Deep Reinforcement Learning- a Continuous Training Pipeline | 2020 | return_target | True |
| W3138331364 | Stock Market Value Prediction using Deep Learning | 2020 | return_target | False |
| W3214241962 | Combined deep learning classifiers for stock market prediction: integrating stock price and news sentiments | 2021 | return_target | False |
| W3156510899 | Harvesting social media sentiment analysis to enhance stock market prediction using deep learning | 2021 | return_target | True |
| W4200347527 | OPINION AND TECHNICAL INDICATOR BASED OPTIMIZED DEEP LEARNING FOR PREDICTION OF STOCK MARKET | 2021 | return_target | True |
| W2902982515 | Stock Return Prediction Using Artificial Neural Networks and Google Search Volumes | 2018 | multi_day | True |
| W4388575137 | Stock Market Prediction Using Deep Reinforcement Learning | 2023 | multi_day | True |
| W4409047305 | Beyond Traditional Models: Revolutionizing Stock Market Prediction with a Deep Learning Framework | 2024 | multi_day | False |
| W4401063612 | Enhancing stock market prediction accuracy with recurrent deep learning models: A case study on the CAC40 index | 2024 | multi_day | True |
| W7236727866 | Advanced Stock Market Prediction Using Deep Learning Algorithms and Optimization Techniques | 2026 | multi_day | False |
| W7144209855 | Deep Learning for Stock Market Prediction: A Systematic Review | 2026 | multi_day | True |
| W7237797617 | Leveraging Machine Learning, Deep Learning, and Ensemble Learning for Feature Selection in Stock Market Prediction | 2026 | multi_day | True |
| W7217348646 | A Technical Analysis Based Prediction of Stock Market Trading Strategies Using Deep Learning and Machine Learning Algorithms | 2022 | leakage | True |
| W7202161411 | A Hybrid Deep Learning Ensemble with Wavelet Feature Extraction for Stock Market Prediction | 2026 | leakage | True |
| W2901852093 | Deep Learning for Stock Market Prediction Using Event Embedding and Technical Indicators | 2018 | news_sentiment | False |
| W2860919791 | Stock market prediction with deep learning using financial news | 2018 | news_sentiment | False |
| W2992709687 | A Deep Learning Model for Dimensional ValenceArousal Intensity Prediction in Stock Market | 2019 | news_sentiment | False |
| W4230316737 | Deep Learning for Stock Market Prediction Using Social Media and Technical Information | 2019 | news_sentiment | True |
| W3008571904 | Deep Learning for the Prediction of Stock Market Trends | 2019 | news_sentiment | True |
| W4248743146 | Hybrid Deep Learning Based Stock Market Prediction with both Sentiment and Historic Trend Data | 2020 | news_sentiment | True |
| W3010063904 | Stock Market Prediction Using Ensemble of Graph Theory, Machine Learning and Deep Learning Models | 2020 | news_sentiment | False |
| W3197302096 | Stock Market Prediction and Investment using Deep Reinforcement Learning- a Continuous Training Pipeline | 2020 | news_sentiment | True |
| W3110233226 | Stock Market Prediction with Deep Learning Using Public Disclosure Platform Data | 2020 | news_sentiment | False |
| W3198938451 | A Proposal for Designing a Deep Learning Model for Analysis and Prediction of Stock Market Movement for Portfolio Management | 2021 | news_sentiment | False |
| W3126088882 | An Empirical Research and Comprehensive Analysis of Stock Market Prediction using Machine Learning and Deep Learning techniques | 2021 | news_sentiment | True |
| W3186683441 | Analysis of Stock Market Prediction Models Using Deep Learning | 2021 | news_sentiment | True |


# kaushiks-2804/FINANCIAL-TIME-SERIES-FORECASTING-FOR-STOCK-PRICES

Source: https://github.com/kaushiks-2804/FINANCIAL-TIME-SERIES-FORECASTING-FOR-STOCK-PRICES
Stars: 1
Updated: 2026-01-05T06:02:25Z

---

# Financial Time Series Forecasting for Stock Prices

Forecast daily stock prices using **LSTM** and **GRU** deep learning models.
This repo rebuilds the project described in your mini-project report (April 2024, Rajalakshmi Engineering College).

## Features
- Load stock data from CSV (downloaded from Yahoo Finance)
- Cleaning & validation (missing values, duplicates)
- Feature engineering: 50/200-day moving averages, RSI
- Normalization (StandardScaler)
- Sequence building for time-series models
- Models: LSTM, GRU (Keras/TensorFlow)
- Evaluation: MAE, MSE, RMSE, R²
- Optional EDA notebook

## Structure
```
financial-time-series-forecasting/
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ main.py
├─ data/
│  └─ README.md
├─ notebooks/
│  └─ exploratory_analysis.ipynb
└─ src/
   ├─ __init__.py
   ├─ data_preprocessing.py
   ├─ feature_engineering.py
   ├─ modeling.py
   └─ evaluation.py
```

## Setup
```bash
git clone https://github.com/<your-username>/financial-time-series-forecasting.git
cd financial-time-series-forecasting
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
1) Put your CSV into `data/stock_data.csv` containing at least: `Date,Open,High,Low,Close,Adj Close,Volume`.
2) Execute:
```bash
python main.py
```

## Notes
- By default the script uses the **Close** column to create sequences.
- If your dataset has multiple tickers, filter to a single symbol first.

## Authors
- Kaushik


# Improving GRU Sentiment Stock Forecasting

Research project on improving GRU-based stock forecasting by incorporating sentiment signals.

> Project mentor: Padmakumari Ma'am

## Overview

This repository contains the scaffolding for experiments that compare baseline GRU
forecasting against improved variants that fuse market sentiment (e.g., derived
from financial news / social media) with price history.

## Status

- [x] Repository scaffold
- [ ] Data collection & preprocessing
- [ ] Baseline GRU model
- [ ] Sentiment integration experiments
- [ ] Evaluation & comparison

## Repository layout

```
.
├── data/           # datasets (raw + processed)
├── src/            # source code
│   ├── baseline.py # baseline GRU forecasting
│   └── experiment.py # sentiment-augmented experiments
├── experiments/    # experiment configs/notes
└── results/        # outputs, metrics, figures
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

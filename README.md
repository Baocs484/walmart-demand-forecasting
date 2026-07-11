# 🏬 Multi-Store Demand Forecasting & Inventory Optimization

[![Tests](https://github.com/Baocs484/walmart-demand-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/Baocs484/walmart-demand-forecasting/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

End-to-end ML system that forecasts weekly sales for **45 Walmart stores × 81 departments** (~421K records) and turns the forecasts into **inventory decisions**: dynamic safety stock (ABC-XYZ policy), restock priorities, and cost simulation.

Built as a portfolio project demonstrating both **data science** (leak-free feature engineering, rolling-origin CV, recursive forecasting, model diagnostics) and **data analysis** (executive dashboard, auto-generated insights, formatted Excel reporting) skills.

## ✨ Highlights

- **7 models compared** — Baseline, RandomForest, XGBoost, GradientBoosting, LightGBM, CatBoost, weight-optimized Ensemble — evaluated on WMAE (holiday weeks ×5, Kaggle standard)
- **Leak-free pipeline** — feature statistics (department aggregates, quantile clipping, NaN-fill values) fitted on the training window only, via a fit/transform design
- **Metric-aligned training** — holiday sample weights (×5), L1 objective, early stopping on a time-based validation split
- **Rolling-origin cross-validation** — 4 folds including the Thanksgiving/Christmas window, because a single test window understates holiday difficulty
- **Recursive multi-week forecasting** — predicts week t+1, feeds it back as a lag, iterates up to 8 weeks
- **Three report surfaces** — analyst dashboard (HTML), model-diagnostics report (HTML), formatted Excel workbook — plus an interactive Streamlit app
- **Engineering hygiene** — 38 unit/integration tests, config-driven parameters, saved pipeline artifacts, run-history tracking, CI

## 📊 Results

Test window = last 15% of the timeline (Jun–Oct 2012), WMAE weighs holiday weeks ×5.

| Model | WMAE ↓ | MAE | MAPE | Train time |
|-------|--------|-----|------|-----------|
| **LightGBM** (default) | **~1,325** | ~1,305 | 13.7% | ~2 min |
| Ensemble (LightGBM+GB) | ~1,330 | ~1,300 | 13.7% | ~8 min |
| CatBoost | ~1,340 | ~1,300 | 13.9% | ~4 min |
| XGBoost | ~1,360 | ~1,310 | 14.7% | ~10 s |
| Baseline (seasonal avg) | 2,780 | 2,693 | 39.8% | <1 s |

→ **~52% WMAE reduction vs. the seasonal baseline**, ~92% forecast accuracy (WAPE-based), total bias ≈ +1%. Numbers above are from the leak-free pipeline; run `python main.py compare` to reproduce the exact table on your machine (saved to `results/model_comparison.csv`).

> **Honest caveat:** the fixed test window contains only Labor Day. Rolling-origin CV (`python validate.py`, 4 folds) shows the fold covering **Thanksgiving + Christmas runs at WMAE ≈ 2,870 — 2.2× harder** than regular-season folds (≈1,270–1,510). Overall CV: WMAE 1,735 ± 768. A single fixed window would have overstated model quality; the CV number is the one to trust.

## 🖥️ Report Surfaces

| Output | Audience | Contents |
|--------|----------|----------|
| `results/dashboard.html` | Business / analysts | KPIs with status colors, auto-generated insights, actual-vs-forecast trend with cluster filter & holiday markers, error attribution by department, ABC-XYZ matrix, restock priorities |
| `results/model_report.html` | Data scientists | Model config + data split, residual distribution & funnel plot, MAE by week, holiday vs regular error, CV per fold, run history, model comparison, feature importance |
| `results/reports/forecast_report.xlsx` | Detailed analysis | 6 formatted sheets: Overview, Model_Comparison, Store_Inventory, Restock_Priority, ABC_XYZ_All, Feature_Importance |
| `streamlit run app.py` | Interactive demo | Store-dept explorer, on-demand recursive forecast, filterable restock table, experiment history |

## 🏗️ Architecture

```
├── main.py                    # CLI: train / compare / forecast
├── compare_models.py          # Detailed 7-model benchmark
├── validate.py                # Rolling-origin cross-validation
├── app.py                     # Streamlit interactive app
├── config.yaml                # Business & pipeline parameters
│
├── src/
│   ├── system.py              # Pipeline orchestrator
│   ├── data_processor.py      # Leak-free fit/transform feature engineering (36 features)
│   ├── forecaster.py          # Recursive multi-week forecasting
│   ├── store_clustering.py    # K-Means store clustering (used as a model feature)
│   ├── metrics.py             # WMAE, MAE, RMSE, MAPE + business metrics
│   ├── inventory_report.py    # ABC-XYZ safety stock + restock recommendations
│   ├── report_builder.py      # Dashboard + model report + Excel builders
│   ├── persistence.py         # Pipeline save/load (joblib)
│   ├── run_history.py         # Lightweight experiment tracking (runs.jsonl)
│   ├── config.py              # config.yaml loader with defaults
│   └── models/                # BaseModel + 7 implementations
│
├── tests/                     # 38 pytest tests (unit + integration)
├── data/{raw,processed}/      # Kaggle CSVs → merged walmart_clean.csv
├── models/                    # Saved pipeline artifacts (gitignored)
└── results/                   # Generated reports (gitignored)
```

## 🚀 Quickstart

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. Data: download from Kaggle and merge
#    https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
#    put train.csv, stores.csv, features.csv into data/raw/, then:
python scripts/merge_walmart.py

# 3. Train + evaluate + build all reports (~2-3 min)
python main.py train

# 4. Everything else
python main.py compare           # benchmark all 7 models (~20 min)
python main.py forecast --weeks 4  # recursive future forecast from saved model
python validate.py               # rolling-origin cross-validation (~10 min)
streamlit run app.py             # interactive app
python -m pytest tests/ -v       # test suite
```

## 🔬 Method Notes

**Features (36).** Calendar + cyclical encodings, sales lags (1/4/52 weeks), rolling means/std over lags, cross-sectional market-share and rank features computed from *lagged* values within each date (known at prediction time), department aggregates, markdown totals, store type/size, and K-Means cluster id.

**Leakage control.** All dataset-level statistics are fitted on the training window only and applied to validation/test via a fit/transform interface. Base features look strictly backward. Splits are time-based with asserted non-overlap.

**Training ↔ metric alignment.** The evaluation metric (WMAE) weighs holiday weeks 5×, so models train with `sample_weight=5` on holiday rows, use L1-family objectives, and early-stop on weighted validation loss. The ensemble optimizes its blend weights against weighted MAE.

**Recursive forecasting.** The models are 1-step-ahead by construction (they consume last week's sales as a lag). `main.py forecast` iterates prediction → feedback → re-featurize to reach longer horizons; uncertainty compounds with horizon, which is reported honestly in the app.

**Inventory policy.** Forecast error std → dynamic safety stock via ABC (revenue share) × XYZ (volatility) service-level targets; series with <4 weeks of data or non-positive sales are excluded. All quantities are in dollars — the public dataset has no unit prices.

## ⚠️ Known Limitations

- Markdown/promotion data is sparse (missing before Nov 2011) — treated as zero when absent.
- Recursive forecasts inherit compounding error; horizons beyond ~8 weeks are not recommended.
- Cost simulation uses illustrative cost parameters (`config.yaml`) — calibrate before real-world use.

## 📝 Credits

- Dataset: [Walmart Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- Authors: Lê Gia Bảo

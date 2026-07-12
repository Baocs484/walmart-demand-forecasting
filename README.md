# 🏬 Multi-Store Demand Forecasting & Inventory Optimization

[![Tests](https://github.com/Baocs484/walmart-demand-forecasting/actions/workflows/tests.yml/badge.svg)](https://github.com/Baocs484/walmart-demand-forecasting/actions)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

End-to-end machine learning system that forecasts weekly sales for **45 Walmart stores × 81 departments** (~421K records, Feb 2010 – Oct 2012) and turns those forecasts into **inventory decisions**: dynamic safety stock via an ABC-XYZ policy, prioritized restock recommendations, and cost simulation across competing policies.

Built as a portfolio project demonstrating both **data science** (leak-free feature engineering, rolling-origin cross-validation, recursive forecasting, model diagnostics) and **data analysis** (executive dashboard, auto-generated insights, formatted Excel reporting) skills.

## ✨ Highlights

- **7 models benchmarked** — Baseline, RandomForest, XGBoost, GradientBoosting, LightGBM, CatBoost, and a weight-optimized Ensemble — on WMAE (holiday weeks ×5, the Kaggle competition standard)
- **Leak-free pipeline** — every dataset-level statistic is fitted on the training window only, through a scikit-learn-style fit/transform design
- **Metric-aligned training** — holiday sample weights (×5), L1 objectives, early stopping on a time-based validation split
- **Rolling-origin cross-validation** — 4 time folds including the Thanksgiving/Christmas window, because a single test window understates holiday difficulty (it does: 2.2×)
- **Recursive multi-week forecasting** — predicts week t+1, feeds it back as a lag, iterates up to 8 weeks into the future
- **Four report surfaces** — analyst dashboard (HTML), model-diagnostics report (HTML), formatted Excel workbook, interactive Streamlit app
- **Engineering hygiene** — 38 tests, config-driven parameters, saved pipeline artifacts, experiment tracking, GitHub Actions CI

## 🎯 The Problem

Given three raw files from the [Kaggle Walmart competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting):

| File | Contents |
|------|----------|
| `train.csv` | Weekly sales per (Store, Dept, Date) + holiday flag — 421,570 rows |
| `stores.csv` | Store type (A/B/C) and physical size |
| `features.csv` | Temperature, fuel price, CPI, unemployment, 5 markdown (promotion) columns |

predict `Weekly_Sales` for every store-department pair, then decide **how much inventory each pair needs**.

What makes it non-trivial:

- **Holiday weeks are weighted 5×** in the evaluation metric (WMAE) but are rare (~7% of weeks) and behave differently — Thanksgiving week sales jump ~27% network-wide
- **3,300+ individual time series** ranging from $67K/week department flagships to near-zero micro-series, including *negative* sales (returns)
- **Markdown data is missing before Nov 2011**, and the two hardest holidays (Thanksgiving, Christmas) appear only twice in the 143-week history
- Sales are in **dollars, not units** — the inventory layer must present values accordingly

## ⚙️ How It Works

```
raw CSVs ──► merge & validate ──► feature engineering (36 features, leak-free)
                                        │
                     time-based split 70/15/15 (statistics fitted on train only)
                                        │
                 K-Means store clustering (5 clusters, fitted on train → model feature)
                                        │
              train with holiday sample-weights ×5 + L1 objective + early stopping
                                        │
        evaluate on test (WMAE/MAE/RMSE/MAPE) ──► ABC-XYZ inventory policy
                                        │
   dashboard.html + model_report.html + forecast_report.xlsx + saved pipeline artifacts
```

**1. Feature engineering (36 features)** — `src/data_processor.py`

| Group | Features | Leak-safety |
|-------|----------|-------------|
| Calendar | Month, Quarter, WeekOfYear, IsWeekend, IsHoliday + sin/cos cyclical encodings | pure calendar |
| Lags | Sales lag 1 / 4 / 52 weeks | backward-looking |
| Rolling | mean(4), mean(13), std(4) over lagged sales | backward-looking |
| Cross-sectional | regional & national market share, store rank, dept share/rank within store, store rank within dept — all computed from **lagged** values within each date | known at prediction time |
| Growth | 4-week and year-over-year growth of lagged sales (clipped) | backward-looking |
| Department stats | dept average/CV/zero-frequency/size share | **fitted on train only** |
| Store & promo | type encoding, size, K-Means cluster id, markdown total & flag | cluster fitted on train |

NaN-fill values (per-series means) and 1%–99% clipping bounds are also fitted on train and applied everywhere — `process_full(df, train_cutoff_date)` for training, `transform(df)` for inference.

**2. Models** — `src/models/`

All models inherit `BaseModel` (shared feature list, holiday weights, non-negative clipping):

| Model | Key config | Notes |
|-------|-----------|-------|
| Baseline | store/dept historical means × monthly seasonal factors | sanity floor |
| RandomForest | 100 trees, depth 12 | |
| XGBoost | ≤2000 trees, `eval_metric=mae`, early stopping 50 | |
| GradientBoosting | sklearn, 100 trees | slowest, weakest booster |
| LightGBM ⭐ | ≤2000 trees, `objective=regression_l1`, early stopping 50 | default — best speed/accuracy |
| CatBoost | ≤2000 iters, `loss=MAE`, categorical Store/Dept/Cluster | |
| Ensemble | LightGBM + GB, blend weight optimized on **weighted** validation MAE (L-BFGS-B) | best WMAE, 4× slower |

**3. Inventory optimization** — `src/inventory_report.py`

- **ABC** classes by cumulative revenue share (A = top 70%, B = next 20%, C = rest); **XYZ** by coefficient of variation (X ≤ 0.2, Y ≤ 0.5, Z > 0.5)
- Safety stock = `z(ABC,XYZ) × σ(forecast error) × √lead_time`, with z from 2.576 (AX, 99.5% service) down to 1.28 (CZ, 90%) — configurable in `config.yaml`
- Restock priority = 70% stockout rate + 30% forecast error; series with <4 weeks of data or non-positive sales (returns) are excluded as statistical noise
- Policy simulation compares flat-95%, flat-98% and the dynamic ABC-XYZ policy on stockout + holding cost

## 📊 Results

Fixed test window = last 15% of the timeline (Jun–Oct 2012), WMAE weighs holiday weeks ×5:

| Model | WMAE ↓ | MAE | MAPE | Train time |
|-------|--------|-----|------|-----------|
| **LightGBM** (default) | **1,325** | 1,305 | 13.7% | ~2 min |
| Ensemble (LightGBM+GB) | ~1,330 | ~1,300 | 13.7% | ~8 min |
| CatBoost | ~1,340 | ~1,300 | 13.9% | ~4 min |
| XGBoost | ~1,360 | ~1,310 | 14.7% | ~10 s |
| Baseline (seasonal avg) | 2,780 | 2,693 | 39.8% | <1 s |

→ **~52% WMAE reduction vs. the seasonal baseline**, ~92% forecast accuracy (WAPE-based), total bias ≈ +1% (within the ±2% planning band). Reproduce with `python main.py compare`.

**Rolling-origin cross-validation** (`python validate.py`) tells the fuller story:

| Fold | Test window | Holiday weeks | WMAE | MAPE |
|------|------------|---------------|------|------|
| 1 | Nov 2011 – Jan 2012 (**Thanksgiving + Christmas**) | 2 | **2,874** | 20.3% |
| 2 | Feb – Apr 2012 (Super Bowl) | 1 | 1,514 | 18.0% |
| 3 | May – Jul 2012 | 0 | 1,282 | 13.6% |
| 4 | Aug – Oct 2012 (Labor Day) | 1 | 1,269 | 13.8% |
| **Mean ± std** | | | **1,735 ± 768** | 16.4 ± 3.3 |

> **Honest caveat:** the fixed test window contains only Labor Day, so it flatters the model. The major-holiday fold is **2.2× harder** than regular-season folds — the CV mean is the number to trust, and holiday forecasting is the clear improvement frontier.

**Recursive forecast sanity check:** forecasting 4 weeks beyond the data's end, the system automatically flags 2012-11-23 as Thanksgiving week and lifts the network forecast to $62.6M vs ~$49M in regular weeks (+27%) — consistent with observed holiday lifts.

## 🖥️ Report Surfaces

| Output | Audience | Contents |
|--------|----------|----------|
| `results/dashboard.html` | Business / analysts | KPIs with status colors, auto-generated key takeaways, actual-vs-forecast trend with cluster filter & holiday markers, error attribution by department, ABC-XYZ matrix with revenue share, restock priority table |
| `results/model_report.html` | Data scientists | Model config + data split, residual distribution & residual-vs-level funnel, MAE by week, holiday vs regular error, CV per fold, run history, model comparison, feature importance |
| `results/reports/forecast_report.xlsx` | Detailed analysis | 6 formatted sheets (headers, freeze panes, filters, conditional color scales): Overview, Model_Comparison, Store_Inventory, Restock_Priority, ABC_XYZ_All, Feature_Importance |
| `streamlit run app.py` | Interactive demo | Network overview, store-dept explorer, on-demand recursive forecast with CSV download, filterable restock table, experiment history |

Both HTML pages are fully self-contained (open offline, email-able) and cross-linked. Raw CSVs in `results/inventory/` feed Power BI / pandas directly.

## 🏗️ Project Structure

```
├── main.py                    # CLI: train / compare / forecast
├── compare_models.py          # Detailed 7-model benchmark
├── validate.py                # Rolling-origin cross-validation
├── app.py                     # Streamlit interactive app
├── config.yaml                # Business & pipeline parameters
│
├── src/
│   ├── system.py              # Pipeline orchestrator
│   ├── data_processor.py      # Leak-free fit/transform feature engineering
│   ├── forecaster.py          # Recursive multi-week forecasting
│   ├── store_clustering.py    # K-Means store clustering
│   ├── metrics.py             # WMAE, MAE, RMSE, MAPE + business metrics
│   ├── inventory_report.py    # ABC-XYZ safety stock + restock recommendations
│   ├── report_builder.py      # Dashboard + model report + Excel builders
│   ├── persistence.py         # Pipeline save/load (joblib)
│   ├── run_history.py         # Experiment tracking (runs.jsonl)
│   ├── config.py              # config.yaml loader with defaults
│   └── models/                # BaseModel + 7 implementations
│
├── tests/                     # 38 pytest tests (unit + integration)
├── .github/workflows/         # CI: pytest on every push/PR
├── data/{raw,processed}/      # Kaggle CSVs → merged walmart_clean.csv (gitignored)
├── models/                    # Saved pipeline artifacts (gitignored)
└── results/                   # Generated reports (gitignored)
```

## 🚀 Quickstart

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. Data: download train.csv, stores.csv, features.csv from Kaggle
#    https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
#    into data/raw/, then:
python scripts/merge_walmart.py
```

| Command | What it does | Time |
|---------|--------------|------|
| `python main.py train` | Train LightGBM, evaluate, build all reports, save pipeline | ~2–3 min |
| `python main.py train --model xgboost` | Same with XGBoost | ~1 min |
| `python main.py compare` | Benchmark all 7 models | ~20 min |
| `python main.py forecast --weeks 4` | Recursive future forecast from the saved pipeline | ~1 min |
| `python validate.py` | Rolling-origin CV (4 folds × 13 weeks) | ~10 min |
| `streamlit run app.py` | Interactive app at localhost:8501 | instant |
| `python -m pytest tests/ -v` | Test suite | ~10 s |

Business parameters (split ratios, holiday weight, service-level z-scores, lead time, cost assumptions, forecast horizon) live in **`config.yaml`** — no code edits needed to tune the policy.

## ⚠️ Known Limitations

- Markdown/promotion data is sparse (absent before Nov 2011) — treated as zero when missing.
- Recursive forecasts compound their own errors; horizons beyond ~8 weeks are not recommended.
- Cost simulation uses illustrative parameters (`config.yaml`) — calibrate before real-world use.
- Only two observations of Thanksgiving/Christmas exist in the data; holiday accuracy would benefit most from more history or holiday-proximity features (roadmap).

## 📝 Credits

- Dataset: [Walmart Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- Authors: Lê Gia Bảo

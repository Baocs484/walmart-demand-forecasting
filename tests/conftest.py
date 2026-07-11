# tests/conftest.py
"""
Shared pytest fixtures for the demand forecasting test suite.

All fixtures generate synthetic Walmart-like data so tests can run
without the real dataset.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path manipulation – ensure ``from src.xxx import ...`` works regardless of
# how pytest is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_walmart_data(
    n_stores: int = 3,
    n_depts: int = 3,
    n_weeks: int = 60,
    start_date: str = "2010-02-05",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a DataFrame that mimics the Walmart dataset schema.

    Columns produced:
        Store, Dept, Date, Weekly_Sales, IsHoliday,
        Type, Size, Temperature, Fuel_Price,
        MarkDown1–5, CPI, Unemployment
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-FRI")

    rows = []
    store_meta = {
        1: {"Type": "A", "Size": 151315},
        2: {"Type": "B", "Size": 202307},
        3: {"Type": "C", "Size": 39690},
    }

    for store in range(1, n_stores + 1):
        meta = store_meta.get(store, {"Type": "B", "Size": 100000})
        for dept in range(1, n_depts + 1):
            base_sales = rng.uniform(5000, 50000)
            for i, date in enumerate(dates):
                # Seasonal component
                month = date.month
                seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * month / 12)

                # Random noise
                noise = rng.normal(0, base_sales * 0.1)

                weekly_sales = max(0, base_sales * seasonal + noise)

                # Holiday flag – mark ~10 % of weeks as holidays
                is_holiday = bool(rng.random() < 0.10)

                # Markdowns – available only after ~30 weeks
                if i >= 30:
                    md1 = rng.uniform(0, 5000)
                    md2 = rng.uniform(0, 3000)
                    md3 = rng.uniform(0, 200)
                    md4 = rng.uniform(0, 4000)
                    md5 = rng.uniform(0, 3000)
                else:
                    md1 = md2 = md3 = md4 = md5 = np.nan

                rows.append(
                    {
                        "Store": store,
                        "Dept": dept,
                        "Date": date,
                        "Weekly_Sales": round(weekly_sales, 2),
                        "IsHoliday": is_holiday,
                        "Type": meta["Type"],
                        "Size": meta["Size"],
                        "Temperature": rng.uniform(30, 100),
                        "Fuel_Price": rng.uniform(2.5, 4.0),
                        "MarkDown1": md1,
                        "MarkDown2": md2,
                        "MarkDown3": md3,
                        "MarkDown4": md4,
                        "MarkDown5": md5,
                        "CPI": rng.uniform(120, 230),
                        "Unemployment": rng.uniform(4, 14),
                    }
                )

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Raw synthetic DataFrame (3 stores × 3 depts × 60 weeks)."""
    return _generate_synthetic_walmart_data()


@pytest.fixture(scope="session")
def processed_df(sample_df):
    """``sample_df`` after full feature engineering via DataProcessor."""
    from src.data_processor import DataProcessor

    dp = DataProcessor()
    df = dp.create_features(sample_df.copy())
    df = dp._handle_missing_values(df)
    df = dp._validate_and_clean(df)
    return df


@pytest.fixture(scope="session")
def train_val_test_split(processed_df):
    """Chronological train / validation / test split (60 / 20 / 20 %).

    Returns a dict with keys ``train``, ``val``, ``test``.
    """
    df = processed_df.copy()
    dates_sorted = sorted(df["Date"].unique())
    n = len(dates_sorted)

    train_end = dates_sorted[int(n * 0.6)]
    val_end = dates_sorted[int(n * 0.8)]

    train = df[df["Date"] <= train_end].copy()
    val = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
    test = df[df["Date"] > val_end].copy()

    return {"train": train, "val": val, "test": test}

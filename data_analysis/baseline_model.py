"""
baseline_model.py

Baseline modelling for:
Lost Customer Hours (London Underground) vs weather (monthly aggregates)

Models:
1) Rainfall only
2) Weather (extended)
3) Weather + seasonality (month dummies) + trend

Input:
- data/processed/tube_weather_monthly.csv

Output:
- Printed metrics + coefficient tables
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


DATA_PATH = "data/processed/tube_weather_monthly.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# Column naming / readability

# Original columns you had: RR_sum, TX_mean, TN_mean, SS_mean, HU_mean
# We rename them to clearer names for reporting.
RENAME_MAP = {
    "RR_sum": "rainfall_mm_sum",
    "TX_mean": "max_temp_mean",      # (check units in your weather dataset docs)
    "TN_mean": "min_temp_mean",      # (check units)
    "SS_mean": "sunshine_mean",      # (check units)
    "HU_mean": "humidity_mean",      # (check units)
}

TARGET_COL = "lost_customer_hours"
DATE_COL = "month"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Version-safe RMSE (avoids sklearn squared=False)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])

    # basic cleaning
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    # rename weather columns if present
    for old, new in RENAME_MAP.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - month_num: 1..12
    - month dummies: month_2..month_12 (drop_first=True)
    - time_index: 0..N-1 (trend)
    """
    out = df.sort_values(DATE_COL).copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL])

    out["month_num"] = out[DATE_COL].dt.month.astype(int)
    dummies = pd.get_dummies(out["month_num"], prefix="month", drop_first=True)
    out = pd.concat([out, dummies], axis=1)

    out["time_index"] = np.arange(len(out), dtype=int)

    return out


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict:
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict + evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    model_rmse = rmse(y_test.values, y_pred)

    # coefficients table
    coef_df = pd.DataFrame(
        {"feature": feature_cols, "coefficient": model.coef_}
    )
    coef_df["abs_coeff"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coeff", ascending=False).drop(columns=["abs_coeff"])

    print(f"\n=== {label} ===")
    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")
    print(f"R² score: {r2:.3f}")
    print(f"RMSE: {model_rmse:,.0f}")
    print(f"Intercept: {model.intercept_:,.0f}")
    print("\nTop coefficients (by absolute value):")
    print(coef_df.head(15).to_string(index=False))

    return {
        "label": label,
        "r2": r2,
        "rmse": model_rmse,
        "intercept": float(model.intercept_),
        "coefficients": coef_df,
        "model": model,
    }


def main() -> None:
    df = load_data(DATA_PATH)

    # --- Model 1: Rainfall only ---
    # support both original and renamed columns
    rainfall_col = "rainfall_mm_sum" if "rainfall_mm_sum" in df.columns else "RR_sum"
    train_and_evaluate(
        df=df,
        feature_cols=[rainfall_col],
        label="Model 1: Rainfall only",
    )

    # --- Model 2: Weather (extended) ---
    # build list based on columns that exist (renamed preferred)
    preferred = ["rainfall_mm_sum", "max_temp_mean", "min_temp_mean", "sunshine_mean", "humidity_mean"]
    fallback = ["RR_sum", "TX_mean", "TN_mean", "SS_mean", "HU_mean"]

    if all(col in df.columns for col in preferred):
        weather_cols = preferred
    elif all(col in df.columns for col in fallback):
        weather_cols = fallback
    else:
        # mixed case: use whatever is available from either set
        weather_cols = [c for c in preferred if c in df.columns]
        if len(weather_cols) == 0:
            weather_cols = [c for c in fallback if c in df.columns]

    train_and_evaluate(
        df=df,
        feature_cols=weather_cols,
        label="Model 2: Weather (extended)",
    )

    # --- Model 3: Weather + seasonality + trend ---
    df_seasonal = add_seasonality_features(df)

    seasonality_cols = [c for c in df_seasonal.columns if c.startswith("month_")]
    feature_cols_3 = weather_cols + ["month_num"] + seasonality_cols + ["time_index"]

    train_and_evaluate(
        df=df_seasonal,
        feature_cols=feature_cols_3,
        label="Model 3: Weather + seasonality (month dummies) + trend",
    )


if __name__ == "__main__":
    main()
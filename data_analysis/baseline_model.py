import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


DATA_PATH = "data/processed/tube_weather_monthly.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def rmse(y_true, y_pred) -> float:
    """Compute RMSE without relying on sklearn's squared=False (version-safe)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, label: str) -> LinearRegression:
    """Train/test split, fit a linear regression model, print metrics, and return the fitted model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    model_rmse = rmse(y_test, y_pred)

    print(f"\n=== {label} ===")
    print(f"Rows: {len(X):,} | Features: {X.shape[1]}")
    print(f"R² score: {r2:.3f}")
    print(f"RMSE: {model_rmse:,.0f}")
    print(f"Intercept: {model.intercept_:,.0f}")

    # coefficients table (sorted by absolute magnitude)
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    })
    coefs["abs_coeff"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coeff", ascending=False).drop(columns=["abs_coeff"])

    print("\nTop coefficients (by absolute value):")
    print(coefs.head(15).to_string(index=False))

    return model


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonality + trend features:
    - month_num: 1..12
    - month one-hot dummies: month_1..month_12 (drop_first=True to avoid collinearity)
    - time_index: 0..N-1 (captures long-term trend)
    """
    out = df.copy()

    # ensure datetime
    out["month"] = pd.to_datetime(out["month"], errors="coerce")
    out = out.dropna(subset=["month"])

    out["month_num"] = out["month"].dt.month
    out["time_index"] = np.arange(len(out))

    month_dummies = pd.get_dummies(out["month_num"], prefix="month", drop_first=True)
    out = pd.concat([out, month_dummies], axis=1)

    return out


if __name__ == "__main__":
    # load
    final_df = pd.read_csv(DATA_PATH, parse_dates=["month"])

    # Basic cleaning
    final_df = final_df.dropna(subset=["lost_customer_hours"]).copy()
    final_df["lost_customer_hours"] = pd.to_numeric(final_df["lost_customer_hours"], errors="coerce")
    final_df = final_df.dropna(subset=["lost_customer_hours"])

    # Model 1: Rainfall only
    X_simple = final_df[["RR_sum"]]
    y = final_df["lost_customer_hours"]
    train_and_evaluate(X_simple, y, label="Model 1: Rainfall only")

    # Model 2: Extended (weather variables)
    features_weather = ["RR_sum", "TX_mean", "TN_mean", "SS_mean", "HU_mean"]
    X_extended = final_df[features_weather]
    train_and_evaluate(X_extended, y, label="Model 2: Weather (extended)")

    # Model 3: Weather + Seasonality (month dummies) + trend
    df_seasonal = add_seasonality_features(final_df)

    month_dummy_cols = [c for c in df_seasonal.columns if c.startswith("month_")]
    features_seasonal = features_weather + ["time_index"] + month_dummy_cols

    X_seasonal = df_seasonal[features_seasonal]
    y_seasonal = df_seasonal["lost_customer_hours"]

    train_and_evaluate(
        X_seasonal,
        y_seasonal,
        label="Model 3: Weather + seasonality (month dummies) + trend"
    )







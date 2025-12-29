# data loading + cleaning + features

import pandas as pd
import numpy as np

tube_path = "data/raw/tfl-tube-performance.xlsx"
weather_path = "data/raw/london_weather_data_1979_to_2023.csv"


def load_data():
    tube_df = pd.read_excel(tube_path, sheet_name="Lost customer hours")
    weather_df = pd.read_csv(weather_path)
    calendar_df = pd.read_excel(tube_path, sheet_name="Key trends")
    return tube_df, weather_df, calendar_df


def tidy_tube_lch(tube_df: pd.DataFrame) -> pd.DataFrame:
    tube_long = tube_df.melt(
        id_vars=["Financial Year"],
        var_name="period",
        value_name="lost_customer_hours",
    )

    tube_long["period"] = tube_long["period"].str.extract(r"(\d+)").astype(int)

    # convert to numeric (some cells can be strings)
    tube_long["lost_customer_hours"] = pd.to_numeric(tube_long["lost_customer_hours"], errors="coerce")

    # drop blanks
    tube_long = tube_long.dropna(subset=["lost_customer_hours"])

    return tube_long


def clean_period_calendar(calendar_df: pd.DataFrame) -> pd.DataFrame:
    cal = calendar_df[
        ["Period and Financial year", "Reporting Period", "Period ending", "Month"]
    ].rename(
        columns={
            "Reporting Period": "period",
            "Period ending": "period_end",
            "Month": "month",
        }
    )

    cal = cal.dropna(subset=["period"])
    cal["period"] = cal["period"].astype(int)

    # Extract financial year token like "11/12" from "02_11/12"
    cal["Financial Year"] = cal["Period and Financial year"].astype(str).str.extract(r"(\d{2}/\d{2})")
    cal["Financial Year"] = "20" + cal["Financial Year"]

    # Convert dates properly
    cal["period_end"] = pd.to_datetime(cal["period_end"], errors="coerce")
    cal["month"] = pd.to_datetime(cal["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    # Drop rows that failed date conversion
    cal = cal.dropna(subset=["Financial Year", "period_end", "month"])

    return cal[["Financial Year", "period", "period_end", "month"]]


def monthly_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    w = weather_df.copy()

    w["date"] = pd.to_datetime(w["DATE"].astype(str), format="%Y%m%d", errors="coerce")
    w = w.dropna(subset=["date"])

    # Coerce key weather columns to numeric (prevents dtype=object mean errors)
    for col in ["TX", "TN", "RR", "SS", "HU"]:
        if col in w.columns:
            w[col] = pd.to_numeric(w[col], errors="coerce")

    w["month"] = w["date"].dt.to_period("M").dt.to_timestamp()

    weather_monthly = (
        w.groupby("month", as_index=False)
        .agg(
            TX_mean=("TX", "mean"),
            TN_mean=("TN", "mean"),
            RR_sum=("RR", "sum"),
            SS_mean=("SS", "mean"),
            HU_mean=("HU", "mean"),
        )
    )

    return weather_monthly


def main():
    tube_df, weather_df, calendar_df = load_data()

    tube_long = tidy_tube_lch(tube_df)
    calendar_clean = clean_period_calendar(calendar_df)
    weather_monthly = monthly_weather(weather_df)

    # merge tube with calendar dates
    tube_with_dates = tube_long.merge(
        calendar_clean,
        on=["Financial Year", "period"],
        how="inner",
    )

    print("\n=== Tube with dates (ready to merge to weather) ===")
    print(tube_with_dates.head())
    print(tube_with_dates.info())

    print("\n=== Weather monthly ===")
    print(weather_monthly.head())
    print(weather_monthly.info())

    # final merge with weather
    final_df = tube_with_dates.merge(weather_monthly, on="month", how="left")

    final_df = final_df.rename(columns={
    "TX_mean": "avg_max_temp",
    "TN_mean": "avg_min_temp",
    "RR_sum": "total_rainfall",
    "SS_mean": "avg_sunshine",
    "HU_mean": "avg_humidity",
})

    print("\n=== Final merged dataset (tube + weather) ===")
    print(final_df.head())
    print(final_df.info())

    final_df.to_csv("data/tube_weather_monthly.csv", index=False)


DATA_PATH = "data/processed/tube_weather_monthly.csv"

def load_processed_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the processed monthly Tube + weather dataset.
    """
    df = pd.read_csv(path, parse_dates=["month"])
    df = df.dropna(subset=["lost_customer_hours"]).copy()
    df["lost_customer_hours"] = pd.to_numeric(
        df["lost_customer_hours"], errors="coerce"
    )
    return df.dropna(subset=["lost_customer_hours"])


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add month-based seasonality features.
    """
    out = df.copy()
    out["month_num"] = out["month"].dt.month

    month_dummies = pd.get_dummies(
        out["month_num"], prefix="month", drop_first=True
    )
    out = pd.concat([out, month_dummies], axis=1)

    return out


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a linear time trend.
    """
    out = df.sort_values("month").copy()
    out["time_index"] = np.arange(len(out))
    return out


from typing import List

def add_lag_features(
    df: pd.DataFrame,
    target: str = "lost_customer_hours",
    lags: List[int] = [1, 3, 6],
) -> pd.DataFrame:
    """
    Add lagged versions of the target variable.
    """
    out = df.sort_values("month").copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out


if __name__ == "__main__":
    main()


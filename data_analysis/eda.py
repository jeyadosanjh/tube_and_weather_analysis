"""
Exploratory Data Analysis (EDA) for:
- Lost Customer Hours (London Underground)
- Monthly aggregated weather variables

Outputs figures to /figures.

Figures produced:
1) LCH time series
2) LCH time series + rolling mean + variability band
3) Rainfall vs LCH scatter + trend line
4) Scatter + trend line for each weather variable vs LCH
5) Correlation heatmap (weather + LCH)
6) Distribution plots for LCH (hist + log-hist)
7) Average LCH by calendar month (bar)
8) Seasonality boxplot of LCH by calendar month
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
PROCESSED_PATH = "data/processed/tube_weather_monthly.csv"
FIG_DIR = Path("figures")

RENAME_MAP = {
    "RR_sum": "rainfall_mm_sum",
    "TX_mean": "max_temp_mean",
    "TN_mean": "min_temp_mean",
    "SS_mean": "sunshine_mean",
    "HU_mean": "humidity_mean",
}


# Data loading / preparation
def ensure_fig_dir(fig_dir: Path = FIG_DIR) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)


def load_processed_data(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the processed Tube + weather dataset and enforce:
    - consistent readable column names
    - correct dtypes
    - sorted by month
    """
    df = pd.read_csv(path, parse_dates=["month"])
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    df = df.sort_values("month").reset_index(drop=True)

    # Ensure numeric types for key columns
    numeric_cols = [
        "lost_customer_hours",
        "rainfall_mm_sum",
        "max_temp_mean",
        "min_temp_mean",
        "sunshine_mean",
        "humidity_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["month", "lost_customer_hours"]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic time-derived columns used for EDA.
    """
    out = df.copy()
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    out["time_index"] = np.arange(len(out))  # simple trend index
    return out


# Helper plotting utils
def save_close(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / fig_name, dpi=300)
    plt.close()


def add_linear_trendline(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Fit y = a*x + b. Returns (a, b).
    """
    x = x.astype(float)
    y = y.astype(float)
    a, b = np.polyfit(x, y, 1)
    return a, b


# EDA Figures
def plot_lch_over_time(df: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(df["month"], df["lost_customer_hours"], linewidth=2)
    plt.xlabel("Month")
    plt.ylabel("Lost Customer Hours")
    plt.title("Lost Customer Hours Over Time (Monthly)")
    save_close("lost_customer_hours_over_time.png")


def plot_lch_with_rolling(df: pd.DataFrame, window: int = 6) -> None:
    """
    Rolling mean (trend smoothing) with +/- 1 rolling std band.
    window=6 means ~6 months.
    """
    s = df.set_index("month")["lost_customer_hours"].sort_index()
    roll_mean = s.rolling(window=window, min_periods=1).mean()
    roll_std = s.rolling(window=window, min_periods=1).std()

    plt.figure(figsize=(11, 5))
    plt.plot(s.index, s.values, linewidth=1.5, alpha=0.6, label="Monthly LCH")
    plt.plot(roll_mean.index, roll_mean.values, linewidth=2.5, label=f"{window}-month rolling mean")

    # variability band (optional but nice)
    upper = roll_mean + roll_std
    lower = roll_mean - roll_std
    plt.fill_between(roll_mean.index, lower.values, upper.values, alpha=0.15, label="±1 rolling std")

    plt.xlabel("Month")
    plt.ylabel("Lost Customer Hours")
    plt.title("Lost Customer Hours with Rolling Trend and Variability")
    plt.legend()
    save_close("lost_customer_hours_rolling.png")


def plot_weather_vs_lch(df: pd.DataFrame, x_col: str, title: str, x_label: str, fname: str) -> None:
    """
    Scatter + linear trend line for a chosen weather feature vs LCH.
    """
    x = df[x_col]
    y = df["lost_customer_hours"]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.65)

    a, b = add_linear_trendline(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, linewidth=2, label=f"Trend: y = {a:.2f}x + {b:,.0f}")
    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel("Lost Customer Hours")
    plt.title(title)
    save_close(fname)


def plot_all_weather_scatters(df: pd.DataFrame) -> None:
    """
    Produce a small set of comparable scatter plots for each weather feature.
    Useful for interpretation beyond the correlation heatmap.
    """
    pairs = [
        ("rainfall_mm_sum", "Monthly Rainfall (mm, sum)", "weather_scatter_rainfall.png"),
        ("max_temp_mean", "Max Temperature (°C * 10, mean)", "weather_scatter_max_temp.png"),
        ("min_temp_mean", "Min Temperature (°C * 10, mean)", "weather_scatter_min_temp.png"),
        ("sunshine_mean", "Sunshine (hours, mean)", "weather_scatter_sunshine.png"),
        ("humidity_mean", "Humidity (% , mean)", "weather_scatter_humidity.png"),
    ]

    for col, xlabel, fname in pairs:
        if col in df.columns:
            plot_weather_vs_lch(
                df=df,
                x_col=col,
                title=f"{xlabel} vs Lost Customer Hours",
                x_label=xlabel,
                fname=fname,
            )


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Matplotlib-only correlation heatmap (no seaborn dependency).
    """
    corr_cols = [
        "lost_customer_hours",
        "rainfall_mm_sum",
        "max_temp_mean",
        "min_temp_mean",
        "sunshine_mean",
        "humidity_mean",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, aspect="auto")
    plt.colorbar(im, shrink=0.85)

    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)

    # annotate
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.title("Correlation Heatmap: Weather Variables and Lost Customer Hours")
    save_close("correlation_heatmap.png")


def plot_lch_distribution(df: pd.DataFrame) -> None:
    """
    Histogram + log histogram (helps if LCH is skewed).
    """
    lch = df["lost_customer_hours"].dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(lch, bins=20, edgecolor="black")
    plt.xlabel("Lost Customer Hours")
    plt.ylabel("Frequency")
    plt.title("Distribution of Monthly Lost Customer Hours")
    save_close("lch_distribution.png")

    # log1p version (handles zeros safely)
    plt.figure(figsize=(7, 5))
    plt.hist(np.log1p(lch), bins=20, edgecolor="black")
    plt.xlabel("log(1 + Lost Customer Hours)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Monthly Lost Customer Hours (log scale)")
    save_close("lch_distribution_log.png")


def plot_average_lch_by_month(df: pd.DataFrame) -> None:
    monthly_avg = (
        df.groupby("month_num")["lost_customer_hours"]
        .mean()
        .reset_index()
        .sort_values("month_num")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(monthly_avg["month_num"], monthly_avg["lost_customer_hours"])
    plt.xlabel("Calendar Month")
    plt.ylabel("Average Lost Customer Hours")
    plt.title("Average Lost Customer Hours by Calendar Month")
    plt.xticks(range(1, 13))
    save_close("avg_lch_by_month.png")


def plot_seasonality_boxplot(df: pd.DataFrame) -> None:
    """
    Boxplot of LCH by month number (visual seasonality + spread).
    """
    # Prepare data in order 1..12
    data = [df.loc[df["month_num"] == m, "lost_customer_hours"].dropna().values for m in range(1, 13)]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=list(range(1, 13)), showfliers=True)
    plt.xlabel("Calendar Month")
    plt.ylabel("Lost Customer Hours")
    plt.title("Seasonality: Lost Customer Hours Distribution by Calendar Month")
    save_close("lch_boxplot_by_month.png")


# Summary stats
def print_summary_statistics(df: pd.DataFrame) -> None:
    print("\n=== Summary statistics ===")
    cols = [
        "lost_customer_hours",
        "rainfall_mm_sum",
        "max_temp_mean",
        "min_temp_mean",
        "sunshine_mean",
        "humidity_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].describe())


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_fig_dir()

    df = load_processed_data(PROCESSED_PATH)
    df = add_time_features(df)

    print_summary_statistics(df)

    # Core time series figures
    plot_lch_over_time(df)
    plot_lch_with_rolling(df, window=6)

    # Relationship figures
    if "rainfall_mm_sum" in df.columns:
        plot_weather_vs_lch(
            df=df,
            x_col="rainfall_mm_sum",
            title="Monthly Rainfall vs Lost Customer Hours",
            x_label="Monthly Rainfall (mm, sum)",
            fname="monthly_rainfall_vs_lch.png",
        )

    plot_all_weather_scatters(df)

    # Correlation + distributions + seasonality
    plot_correlation_heatmap(df)
    plot_lch_distribution(df)
    plot_average_lch_by_month(df)
    plot_seasonality_boxplot(df)

    print(f"EDA figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()

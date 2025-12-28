import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

final_df = pd.read_csv(
    "data/processed/tube_weather_monthly.csv",
    parse_dates=["month"]
)

#styling 
sns.set(style="whitegrid")

DATA_PATH = "data/processed/tube_weather_monthly.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["month"])

print(df.head())
print(df.info())

#monthly rainfall vs lost customer hours
plt.figure()

sns.scatterplot(
    data=final_df,
    x="RR_sum",
    y="lost_customer_hours",
    alpha=0.7
)

plt.title("Monthly Rainfall vs Lost Customer Hours on the London Underground")
plt.xlabel("Total Monthly Rainfall (mm)")
plt.ylabel("Lost Customer Hours")

plt.tight_layout()
plt.savefig("monthly_rainfall_vs_lost_customer_hours.png", dpi=300)
plt.show()

#lost customer hours over time
plt.figure()

plt.plot(
    final_df["month"],
    final_df["lost_customer_hours"]
)

plt.title("Lost Customer Hours Over Time")
plt.xlabel("Year")
plt.ylabel("Lost Customer Hours")

plt.tight_layout()
plt.savefig("lost_customer_hours_over_time.png", dpi=300)
plt.show()

#correlation heatmap (weather vs disruption)
corr_df = final_df[
    ["lost_customer_hours", "TX_mean", "TN_mean", "RR_sum", "SS_mean", "HU_mean"]
]

corr = corr_df.corr()
plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Weather Variables and Lost Customer Hours")

plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()


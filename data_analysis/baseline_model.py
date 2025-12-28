from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

final_df = pd.read_csv(
    "data/processed/tube_weather_monthly.csv",
    parse_dates=["month"]
)

# prepare data for modeling
X = final_df[["RR_sum"]]
y = final_df["lost_customer_hours"]

# train/test split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:,.0f}")

print(f"Intercept: {model.intercept_:,.0f}")
print(f"Rainfall coefficient: {model.coef_[0]:,.0f}")

# extended model
features = ["RR_sum", "TX_mean", "TN_mean", "SS_mean", "HU_mean"]

X = final_df[features]
y = final_df["lost_customer_hours"]

# train/test split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#fit the model
from sklearn.linear_model import LinearRegression

extended_model = LinearRegression()
extended_model.fit(X_train, y_train)

# make predictions
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

y_pred_ext = extended_model.predict(X_test)

# evaluate the model
r2_ext = r2_score(y_test, y_pred_ext)
rmse_ext = np.sqrt(mean_squared_error(y_test, y_pred_ext))

print(f"Extended model R²: {r2_ext:.3f}")
print(f"Extended model RMSE: {rmse_ext:,.0f}")

# coefficients
coefficients = pd.DataFrame({
    "feature": features,
    "coefficient": extended_model.coef_
})

print(coefficients)






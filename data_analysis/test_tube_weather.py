import unittest
import pandas as pd
import numpy as np

from tube_weather import (
    tidy_tube_lch,
    clean_period_calendar,
    monthly_weather,
    add_seasonality,
    add_trend,
    add_lag_features,
)

class unit_tests(unittest.TestCase):

    # tidy_tube_lch tests
    def test_tidy_tube_lch_drops_nonnumeric_and_nans(self):
        tube_df = pd.DataFrame({
            "Financial Year": ["2011/12", "2011/12", "2011/12"],
            "P01": [10, "20", "not_a_number"],
            "P02": [None, 5, 7],
        })

        result = tidy_tube_lch(tube_df)

        # valid numeric values should be: 10, 20, 5, 7 (4 rows)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 4)

        # periods should be 1 and 2 only
        self.assertSetEqual(set(result["period"].unique()), {1, 2})

        # lost_customer_hours should be numeric and contain expected values
        self.assertTrue(pd.api.types.is_numeric_dtype(result["lost_customer_hours"]))
        vals = sorted(result["lost_customer_hours"].astype(float).tolist())
        self.assertEqual(vals, [5.0, 7.0, 10.0, 20.0])

    def test_tidy_tube_lch_raises_without_financial_year(self):
        tube_df = pd.DataFrame({
            "P01": [10, 20],
            "P02": [5, 7],
        })
        with self.assertRaises(KeyError):
            tidy_tube_lch(tube_df)


    # clean_period_calendar tests
    def test_clean_period_calendar_types_and_financial_year(self):
        calendar_df = pd.DataFrame({
            "Period and Financial year": ["02_11/12", "03_11/12"],
            "Reporting Period": [2, 3],
            "Period ending": ["2011-05-28", "2011-06-25"],
            "Month": ["2011-05-15", "2011-06-20"],  # not month-start on purpose
        })

        result = clean_period_calendar(calendar_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 2)

        # Financial Year should be created as "2011/12"
        self.assertListEqual(result["Financial Year"].tolist(), ["2011/12", "2011/12"])

        # period should be int
        self.assertTrue(pd.api.types.is_integer_dtype(result["period"]))
        self.assertListEqual(result["period"].tolist(), [2, 3])

        # month and period_end should be datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["month"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["period_end"]))

        # month should be normalised to month-start
        self.assertEqual(result.loc[0, "month"], pd.Timestamp("2011-05-01"))
        self.assertEqual(result.loc[1, "month"], pd.Timestamp("2011-06-01"))

    def test_clean_period_calendar_drops_missing_period(self):
        calendar_df = pd.DataFrame({
            "Period and Financial year": ["02_11/12", "03_11/12"],
            "Reporting Period": [2, np.nan],
            "Period ending": ["2011-05-28", "2011-06-25"],
            "Month": ["2011-05-01", "2011-06-01"],
        })

        result = clean_period_calendar(calendar_df)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.iloc[0]["period"], 2)


    # monthly_weather tests
    def test_monthly_weather_aggregations(self):
        weather_df = pd.DataFrame({
            "DATE": [20200101, 20200102, 20200201],
            "RR": [5.0, 3.0, 2.0],
            "TX": [10.0, 12.0, 8.0],
            "TN": [2.0, 3.0, 1.0],
            "SS": [4.0, 5.0, 6.0],
            "HU": [80.0, 82.0, 78.0],
        })

        result = monthly_weather(weather_df)

        # should produce Jan and Feb rows
        self.assertEqual(result.shape[0], 2)
        self.assertIn("month", result.columns)
        self.assertIn("RR_sum", result.columns)
        self.assertIn("TX_mean", result.columns)

        jan = result[result["month"] == pd.Timestamp("2020-01-01")].iloc[0]
        feb = result[result["month"] == pd.Timestamp("2020-02-01")].iloc[0]

        # Jan rainfall sum = 8
        self.assertAlmostEqual(float(jan["RR_sum"]), 8.0, places=6)
        # Jan TX mean = 11
        self.assertAlmostEqual(float(jan["TX_mean"]), 11.0, places=6)
        # Feb rainfall sum = 2
        self.assertAlmostEqual(float(feb["RR_sum"]), 2.0, places=6)

    def test_monthly_weather_handles_numeric_strings(self):
        weather_df = pd.DataFrame({
            "DATE": [20200101, 20200102],
            "RR": ["5.0", "3.0"],   # strings
            "TX": ["10.0", "12.0"],
            "TN": ["2.0", "3.0"],
            "SS": ["4.0", "5.0"],
            "HU": ["80.0", "82.0"],
        })

        # if your monthly_weather coerces to numeric, this should pass
        result = monthly_weather(weather_df)
        jan = result[result["month"] == pd.Timestamp("2020-01-01")].iloc[0]
        self.assertAlmostEqual(float(jan["RR_sum"]), 8.0, places=6)

    # Feature engineering tests
    def test_add_seasonality_creates_month_num_and_dummies(self):
        df = pd.DataFrame({
            "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "lost_customer_hours": [100.0, 110.0, 120.0],
        })

        out = add_seasonality(df)

        self.assertIn("month_num", out.columns)
        self.assertListEqual(out["month_num"].tolist(), [1, 2, 3])

        # With drop_first=True, month_2 and month_3 should exist, month_1 should not
        self.assertNotIn("month_1", out.columns)
        self.assertIn("month_2", out.columns)
        self.assertIn("month_3", out.columns)

        # Check dummy correctness
        self.assertEqual(int(out.loc[0, "month_2"]), 0)
        self.assertEqual(int(out.loc[1, "month_2"]), 1)

    def test_add_trend_creates_time_index(self):
        df = pd.DataFrame({
            "month": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
            "lost_customer_hours": [120.0, 100.0, 110.0],
        })

        out = add_trend(df)

        # should be sorted by month and time_index sequential
        self.assertListEqual(out["month"].tolist(), list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])))
        self.assertListEqual(out["time_index"].tolist(), [0, 1, 2])

    def test_add_lag_features_values(self):
        df = pd.DataFrame({
            "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "lost_customer_hours": [10.0, 20.0, 30.0, 40.0],
        })

        out = add_lag_features(df, lags=[1, 3])

        self.assertIn("lost_customer_hours_lag_1", out.columns)
        self.assertIn("lost_customer_hours_lag_3", out.columns)

        # lag 1: first is NaN, then previous values
        self.assertTrue(pd.isna(out.loc[0, "lost_customer_hours_lag_1"]))
        self.assertEqual(out.loc[1, "lost_customer_hours_lag_1"], 10.0)
        self.assertEqual(out.loc[3, "lost_customer_hours_lag_1"], 30.0)

        # lag 3: first 3 are NaN, 4th is first value
        self.assertTrue(pd.isna(out.loc[0, "lost_customer_hours_lag_3"]))
        self.assertTrue(pd.isna(out.loc[1, "lost_customer_hours_lag_3"]))
        self.assertTrue(pd.isna(out.loc[2, "lost_customer_hours_lag_3"]))
        self.assertEqual(out.loc[3, "lost_customer_hours_lag_3"], 10.0)


if __name__ == "__main__":
    unittest.main()

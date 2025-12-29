#unit testing for data cleaning functions in tube_weather.py

import unittest
import pandas as pd
from tube_weather import (
    load_data,
    tidy_tube_lch,
    clean_period_calendar,
    monthly_weather,
)

class unit_tests(unittest.TestCase):

    def test_tidy_tube_lch_basic(self):
        # small sample mirroring TfL sheet shape
        tube_df = pd.DataFrame({
            "Financial Year": ["2011/12", "2011/12"],
            "P01": [10, "20"],
            "P02": [None, 5],
        })

        result = tidy_tube_lch(tube_df)

        # result should be a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # 3 valid values (10, 20, 5)
        self.assertEqual(result.shape[0], 3)

        # periods extracted correctly
        self.assertSetEqual(set(result["period"].unique()), {1, 2})

        # lost_customer_hours numeric
        self.assertTrue(
            pd.api.types.is_numeric_dtype(result["lost_customer_hours"])
        )

        self.assertEqual(
            sorted(result["lost_customer_hours"].astype(float).tolist()),
            [5.0, 10.0, 20.0],
        )

    def test_tidy_period_calendar_basic(self):
        calendar_df = pd.DataFrame({
            "Period and Financial year": ["01_2011/12", "02_2011/12"],
            "Reporting Period": [1, 2],
            "Period ending": ["2011-04-30", "2011-05-28"],
            "Month": ["2011-04-01", "2011-05-01"],
        })

        result = clean_period_calendar(calendar_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 2)

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["month"])
        )

        self.assertListEqual(result["period"].tolist(), [1, 2])

    def test_weather_to_monthly_basic(self):
        weather_df = pd.DataFrame({
            "DATE": [20200101, 20200102, 20200201],
            "RR": [5.0, 3.0, 2.0],
            "TX": [10.0, 12.0, 8.0],
            "TN": [2.0, 3.0, 1.0],
            "SS": [4.0, 5.0, 6.0],
            "HU": [80.0, 82.0, 78.0],
        })

        result = monthly_weather(weather_df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], 2)

        self.assertIn("RR_sum", result.columns)
        self.assertIn("TX_mean", result.columns)
        self.assertIn("month", result.columns)


if __name__ == "__main__":
    unittest.main()

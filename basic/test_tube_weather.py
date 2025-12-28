import unittest
import pandas as pd
from tube_weather import tidy_tube_lch

class unit_tests(unittest.TestCase):
	def test_tidy_tube_lch_basic(self):
		# small sample that mirrors the expected sheet shape
		tube_df = pd.DataFrame({
			"Financial Year": ["2011/12", "2011/12"],
			"P01": [10, "20"],
			"P02": [None, 5],
		})

		result = tidy_tube_lch(tube_df)

		# result should be a DataFrame with 3 non-null lost_customer_hours values
		self.assertIsInstance(result, pd.DataFrame)
		self.assertEqual(result.shape[0], 3)

		# periods should have been extracted as integers 1 and 2
		self.assertSetEqual(set(result["period"].unique()), {1, 2})

		# lost_customer_hours should be numeric and contain the expected values
		self.assertTrue(pd.api.types.is_numeric_dtype(result["lost_customer_hours"]))
		self.assertEqual(sorted(result["lost_customer_hours"].astype(float).tolist()), [5.0, 10.0, 20.0])
    

if __name__ == "__main__":
    unittest.main()

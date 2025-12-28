import unittest
from basic_function import tidy_tube_lch

class TestBasicFunction(unittest.TestCase):
    def test_tidy_tube_lch(self):
        # Sample input DataFrame
        import pandas as pd

        data = {
            "Station": ["A", "B"],
            "LCH_2020-01": [100, 200],
            "LCH_2020-02": [150, 250],
        }
        tube_df = pd.DataFrame(data)

        # Expected output DataFrame
        expected_data = {
            "Station": ["A", "A", "B", "B"],
            "period": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "LCH": [100, 150, 200, 250],
        }
        expected_df = pd.DataFrame(expected_data)

        # Run the function
        result_df = tidy_tube_lch(tube_df)

        # Reset index for comparison
        result_df = result_df.reset_index(drop=True)
        expected_df = expected_df.reset_index(drop=True)

        # Assert DataFrames are equal
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == "__main__":
    unittest.main()

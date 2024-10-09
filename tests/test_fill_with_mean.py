import unittest

import numpy as np
import pandas as pd

from wod_predictor.feature_engineering_parts.helpers import fill_missing_values


class TestFillWithMean(unittest.TestCase):
    def test_fill_with_mean_no_missing(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
        result = fill_missing_values(df, method='mean')
        pd.testing.assert_frame_equal(result, df)

    def test_fill_with_mean_some_missing(self):
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [np.nan, np.nan, 3, 2, 1]
        }, dtype=float)
        expected = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 2, 3, 2, 1]
        }, dtype=float)
        result = fill_missing_values(df, method='mean')
        pd.testing.assert_frame_equal(result, expected)

    def test_fill_with_mean_all_missing(self):
        df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan]
        })
        expected = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan]
        })
        result = fill_missing_values(df, method='mean')
        pd.testing.assert_frame_equal(result, expected)

    def test_fill_with_mean_empty_dataframe(self):
        df = pd.DataFrame({
            'A': [],
            'B': []
        })
        result = fill_missing_values(df, method='mean')
        pd.testing.assert_frame_equal(result, df)

    def test_fill_with_mean_single_column(self):
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5]
        })
        expected = pd.DataFrame({
            'A': [1, 2, 3., 4, 5]
        })
        result = fill_missing_values(df, method='mean')
        pd.testing.assert_frame_equal(result, expected)

if __name__ == '__main__':
    unittest.main()

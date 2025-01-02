import unittest
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn import preprocessing

from wod_predictor.feature_engineering_parts.utils.normalization import \
    GenericSklearnScaler


class TestGenericSklearnScaler(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame(
            {
                "name": [
                    "John Smith",
                    "Emma Wilson",
                    "Michael Brown",
                    "Sarah Davis",
                    "James Miller",
                ],
                "Back Squat": [225.5, 185.0, 315.0, 165.5, 275.0],
                "Chad1000x": [45.5, 38.2, 52.3, 41.7, 48.9],
                "Clean and Jerk": [155.0, 135.5, 225.0, 125.0, 185.5],
                "Deadlift": [315.0, 255.5, 405.0, 235.0, 365.0],
                "Fight Gone Bad": [325.0, 298.0, 387.0, 276.0, 342.0],
            }
        )
        self.test_data.index = range(1, len(self.test_data) + 1)
        self.test_data.index.name = "id"
        self.numeric_data = self.test_data.drop("name", axis=1)

    def test_initialization(self):
        scaler = GenericSklearnScaler("StandardScaler")
        self.assertIsInstance(scaler.scaler, preprocessing.StandardScaler)

        scaler = GenericSklearnScaler("MinMaxScaler", feature_range=(0, 2))
        self.assertEqual(scaler.scaler.feature_range, (0, 2))

        with self.assertRaises(ValueError):
            GenericSklearnScaler("NonExistentScaler")

    def test_fit(self):
        scaler = GenericSklearnScaler("StandardScaler")
        scaler.fit(self.numeric_data)

        self.assertTrue(len(scaler.reverse_mapping) > 0)
        for col in self.numeric_data.columns:
            self.assertIn(col, scaler.reverse_mapping)

    def test_transform(self):
        scaler = GenericSklearnScaler("StandardScaler")
        scaler.fit(self.numeric_data)
        transformed_data = scaler.transform(self.numeric_data)

        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(
            list(transformed_data.columns), list(self.numeric_data.columns)
        )
        self.assertTrue(transformed_data.index.equals(self.numeric_data.index))

        for col in transformed_data.columns:
            self.assertAlmostEqual(transformed_data[col].mean(), 0, places=10)
            self.assertAlmostEqual(transformed_data[col].std(ddof=0), 1, places=10)

    def test_fit_transform(self):
        scaler = GenericSklearnScaler("StandardScaler")
        transformed_data = scaler.fit_transform(self.numeric_data)

        self.assertTrue(len(scaler.reverse_mapping) > 0)

        scaler2 = GenericSklearnScaler("StandardScaler")
        expected_data = scaler2.fit(self.numeric_data).transform(self.numeric_data)
        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_reverse(self):
        scaler = GenericSklearnScaler("StandardScaler")
        transformed_data = scaler.fit_transform(self.numeric_data)
        reversed_data = scaler.reverse(transformed_data)

        pd.testing.assert_frame_equal(
            self.numeric_data, reversed_data, check_exact=False, rtol=1e-10
        )
        self.assertTrue(reversed_data.index.equals(self.numeric_data.index))

    def test_series_handling(self):
        scaler = GenericSklearnScaler("StandardScaler")
        series_data = self.numeric_data["Back Squat"]

        transformed_series = scaler.fit_transform(series_data)
        self.assertIsInstance(transformed_series, pd.Series)

        reversed_series = scaler.reverse(transformed_series)
        self.assertIsInstance(reversed_series, pd.Series)
        pd.testing.assert_series_equal(
            series_data, reversed_series, check_exact=False, rtol=1e-10
        )

    def test_different_scalers(self):
        test_scalers = [
            ("StandardScaler", {}),
            ("MinMaxScaler", {"feature_range": (0, 1)}),
            ("RobustScaler", {}),
            ("MaxAbsScaler", {}),
            ("QuantileTransformer", {"n_quantiles": 100}),
        ]

        for scaler_name, kwargs in test_scalers:
            scaler = GenericSklearnScaler(scaler_name, **kwargs)
            transformed = scaler.fit_transform(self.numeric_data)
            reversed_data = scaler.reverse(transformed)

            pd.testing.assert_frame_equal(
                self.numeric_data, reversed_data, check_exact=False, rtol=1e-10
            )

    def test_empty_df(self):
        scaler = GenericSklearnScaler("StandardScaler")
        empty_df = pd.DataFrame(columns=self.numeric_data.columns)
        with self.assertRaises(Exception):
            scaler.fit(empty_df)

    def test_single_column(self):
        scaler = GenericSklearnScaler("StandardScaler")
        single_col = self.numeric_data[["Back Squat"]]
        scaler.fit(single_col)
        transformed = scaler.transform(single_col)
        self.assertEqual(transformed.shape[1], 1)

    def test_single_row(self):
        scaler = GenericSklearnScaler("StandardScaler")
        single_row = self.numeric_data.iloc[[0]]
        scaler.fit(single_row)
        transformed = scaler.transform(single_row)
        self.assertEqual(len(transformed), 1)

    def test_non_numeric_columns(self):
        data = pd.DataFrame(
            {"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [4.0, 5.0, 6.0]}
        )
        scaler = GenericSklearnScaler("StandardScaler")
        scaler.fit(data)
        transformed = scaler.transform(data)
        self.assertEqual(set(transformed.columns), {"A", "B", "C"})

        # Transformed cols have mu=0, std=0
        for col in ["A", "C"]:
            mean = transformed[col].mean()
            self.assertAlmostEqual(mean, 0, places=7)
            std = transformed[col].std(ddof=0)
            self.assertAlmostEqual(std, 1, places=7)

        # Non-numeric cols are unchanged
        pd.testing.assert_series_equal(
            transformed["B"], data["B"], check_dtype=True, check_exact=True
        )


if __name__ == "__main__":
    unittest.main()

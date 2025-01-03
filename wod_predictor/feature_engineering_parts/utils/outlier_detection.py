import numpy as np
import pandas as pd

from wod_predictor.feature_engineering_parts.base import TransformerMixIn


class IQRoutlierDetector(TransformerMixIn):
    def __init__(self):
        self.upper_thresholds = {}
        self.lower_thresholds = {}
        super().__init__()

    def fit(self, df, score_headers=None):
        df_modified = df.copy()

        # If score_headers is None, use all columns thst are numeric
        if score_headers is None:
            score_headers = df.select_dtypes(include=["int", "float"]).columns

        # Sanity check to confirm that the columns are numeric
        for col in score_headers:
            if df_modified[col].dtype not in [int, float]:
                raise ValueError(
                    f"Column {col} is not numeric, convert to numeric first"
                )

        # fill 0 with na first to prevent skewing the results
        df_modified = df_modified.replace(0, np.nan)

        upper_quartiles = df_modified[score_headers].quantile(0.75)
        lower_quartiles = df_modified[score_headers].quantile(0.25)

        iqr = upper_quartiles - lower_quartiles

        for col in score_headers:
            self.upper_thresholds[col] = upper_quartiles[col] + 1.5 * iqr[col]
            self.lower_thresholds[col] = lower_quartiles[col] - 1.5 * iqr[col]
        super().fit()

    def transform(self, df):
        # find interquartile range
        self.check_fit(df=df)

        df_modified = df.copy()

        for col in self.upper_thresholds:
            df_modified.loc[df_modified[col] > self.upper_thresholds[col], col] = np.nan
            df_modified.loc[df_modified[col] < self.lower_thresholds[col], col] = np.nan

        return df_modified

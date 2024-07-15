import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils.validation import check_is_fitted
from ..constants import Constants as c


class BaseScaler(ABC):
    def __init__(self):
        self.scaler = None

    def transform(self, df):
        check_is_fitted(self.scaler)
        columns = self.scaler.feature_names_in_
        transformed_df = df.copy()

        # add missing columns to prevent raising error
        for col in columns:
            if col not in transformed_df.columns:
                transformed_df[col] = np.nan
        transformed_df.loc[:, columns] = self.scaler.transform(transformed_df[columns])
        return transformed_df

    @abstractmethod
    def reverse(self, df):
        pass


class QuantileScaler(BaseScaler):
    def __init__(self, **kwargs):
        self.reverse_mapping = {}
        self.scaler = QuantileTransformer(**kwargs)

    def fit(self, df):
        # rank the data
        self.scaler.fit(df)
        for i, col in enumerate(df.columns):
            self.reverse_mapping[col] = self.scaler.quantiles_[:, i]

    def transform(self, df):
        return super().transform(df)

    def reverse(self, ranked_data):
        """Reverses to original data range (note it will not be exactly the same as the original data)"""
        unranked_data = ranked_data.copy()
        if self.reverse_mapping is None:
            raise ValueError(
                "reverse_mapping is not set. Please run transform method first."
            )
        for col in unranked_data.columns:
            col_clean = col.replace(c.workout_col_prefix, "")
            if col_clean not in self.reverse_mapping:
                raise KeyError(f"Column {col} not found in reverse mapping")
            unranked_data[col] = self.scaler._transform_col(
                unranked_data[col], self.reverse_mapping[col_clean], inverse=True
            )
        return unranked_data


class StandardScalerByWod(BaseScaler):
    def __init__(self):
        self.scaler = StandardScaler()
        self.reverse_mapping = {}

    def fit(self, df):
        """
        Fit the scaler to the data.
        """
        self.scaler.fit(df)

        for i, col in enumerate(df.columns):
            self.reverse_mapping[col] = (self.scaler.mean_[i], self.scaler.scale_[i])

    def transform(self, df):
        """
        Transform the data to have a mean of 0 and a standard deviation of 1.
        """
        return super().transform(df)

    def reverse(self, df):
        """
        Reverse the data transformation to its original scale. We may have a melted dataframe thats a series, so might be good to support that as well.
        """
        df = df.copy()
        if self.reverse_mapping is None:
            raise ValueError(
                "reverse_mapping is not set. Please run transform method first."
            )
        else:
            columns = df.columns
            for col in columns:
                col_clean = col.replace(c.workout_col_prefix, "")
                if col_clean not in self.reverse_mapping:
                    raise KeyError(f"Column {col_clean} is not in the reverse mapping.")
                mean, scale = self.reverse_mapping[col_clean]
                df[col] = df[col] * scale + mean

        return df

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from wod_predictor.feature_engineering_parts.base import TransformerMixIn


class MissingValueImputation(TransformerMixIn):
    """
    Class is used to handle missing values in features. Contains imputation methods
    """

    def __init__(self, method, **kwargs):
        self.method = method
        self._validate_method()
        self.kwargs = kwargs
        super().__init__()

    def _validate_method(self):
        SUPPORTED_METHODS = ["knn", "zero", "mean", "median"]
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Method {self.method} is not supported for fill_missing_values. Please use one of the following: {SUPPORTED_METHODS}"
            )

    def transform(self, df):
        """
        This function will fill in missing values using the KNN algorithm. Assumes the dataset is already cleaned.

        Parameters
        ----------
        neighbors -> number of neighbors to compare to for KNN algorithm: should be (1, 20) inclusive
        identifier_columns -> list of column headers related to the athlete's identity (index, ID, name)
        data_columns -> list of column headers that contain athletes' data
        """
        self.check_fit(df=df)

        if self.method == "knn":
            df_filled = self.fill_missing_knn(df, **self.kwargs)
            return df_filled
        if self.method == "zero":
            return df.fillna(0)
        if self.method == "mean":
            return df.fillna(df.mean())
        if self.method == "median":
            return df.fillna(df.median())

    def fill_missing_knn(self, df, neighbors, data_columns=[]):
        """
        This function will fill in missing values using the KNN algorithm. Assumes outliers are removed.
        """
        if (not isinstance(neighbors, int)) or neighbors <= 0 or neighbors > 20:
            raise Exception("Invalid neighbor argument")

        if not isinstance(data_columns, list):
            raise Exception("Invalid data_columns argument")

        if len(data_columns) == 0:
            # include numeric columns only
            data_columns = df.select_dtypes(include=["int", "float"]).columns

        # must have at least one non-missing value in the column
        data_columns = [col for col in data_columns if df[col].notnull().sum() > 0]

        try:
            df_identifiers = df.drop(columns=data_columns)
            df_modify = df[data_columns]
        except KeyError as e:
            raise KeyError(
                f"A column header in identifier_columns or data_columns is not in df: {e}"
            )

        df_modify = df_modify.fillna(value=np.nan)
        df_modify = df_modify[:].values

        imputer = KNNImputer(n_neighbors=neighbors)
        df_KNN = imputer.fit_transform(df_modify)
        df_KNN = pd.DataFrame(df_KNN, columns=data_columns, index=df.index)

        df_KNN = pd.concat([df_identifiers, df_KNN], axis=1)
        return df_KNN

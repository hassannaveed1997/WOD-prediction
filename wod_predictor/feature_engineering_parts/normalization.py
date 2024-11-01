import importlib
import inspect
from typing import Union, List
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from ..constants import Constants as c


class BaseScaler(ABC):
    def __init__(self):
        self.scaler = None

    def pad_columns(self, df, full_cols=None):
        """Adds missing columns to prevent raising error"""
        if full_cols is None:
            full_cols = self.scaler.feature_names_in_
        for col in full_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def transform(self, df):
        check_is_fitted(self.scaler)
        columns = self.scaler.feature_names_in_
        transformed_df = df.copy()

        # add missing columns to prevent raising error
        transformed_df = self.pad_columns(transformed_df, columns)
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

class GenericSklearnScaler(BaseScaler):
    def __init__(self, scaler_name: str, **kwargs):
        self.scaler = self._instantiate_scaler(scaler_name, **kwargs)
        self.reverse_mapping = {}
        self.is_series = False
        self.series_name = None
        self.numeric_columns = None
        self.non_numeric_columns = None

    def fit(
        self,
        data: Union[pd.DataFrame, pd.Series]
    ) -> 'GenericSklearnScaler':
        """
        Fit the scaler and store transformation parameters.
        Supports both DataFrame and Series inputs.
        """
        df = self._series2df(data)
        self.numeric_columns = self._get_numeric_columns(df)
        self.non_numeric_columns = df.columns.difference(self.numeric_columns).tolist()
        
        if not self.numeric_columns:
            raise ValueError("No numeric columns found in the input data")
        
        self.scaler.fit(df[self.numeric_columns])
        self._store_params(df[self.numeric_columns])
        return self

    def transform(
        self,
        data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform the data using the fitted scaler.
        Supports both DataFrame and Series inputs.
        """
        check_is_fitted(self.scaler)
        df = self._series2df(data)
        original_col_order = df.columns.tolist()
        
        numeric_cols = [col for col in self.numeric_columns if col in df.columns]
        non_numeric_cols = [col for col in self.non_numeric_columns if col in df.columns]
        
        if not numeric_cols:
            raise ValueError("None of the fitted numeric columns found in input data")
        
        full_cols = list(self.reverse_mapping.keys())
        numeric_df = self.pad_columns(df[numeric_cols], full_cols=full_cols)
        arr_result = self.scaler.transform(numeric_df[full_cols])
        result_df = pd.DataFrame(arr_result, columns=full_cols, index=df.index)
        
        # Concat with non-numeric columns
        if non_numeric_cols:
            result_df = pd.concat([
                result_df,
                df[non_numeric_cols]
            ], axis=1)
        
        return self._input2orig_fmt(result_df[full_cols + non_numeric_cols])
    
    def fit_transform(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Fit the scaler to the data and transform it.
        Supports both DataFrame and Series inputs.
        """
        return self.fit(data).transform(data)

    def reverse(
        self,
        data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Reverse the data transformation to its original scale.
        Only reverses numeric columns, preserving non-numeric ones.
        """
        df = self._series2df(data)
        renamed_cols = [col.replace(c.workout_col_prefix, "") for col in df.columns]
        df.columns = renamed_cols
        original_col_order = df.columns.tolist()
        
        if not hasattr(self.scaler, 'inverse_transform'):
            raise ValueError("Scaler does not support inverse transform")
            
        numeric_cols = [col for col in self.numeric_columns if col in df.columns]
        non_numeric_cols = [col for col in self.non_numeric_columns if col in df.columns]

        if not numeric_cols:
            raise ValueError("None of the fitted numeric columns found in input data")
        
        # Transform numeric columns
        transform_df = df[numeric_cols]
        renamed_cols = [col.replace(c.workout_col_prefix, "") for col in transform_df.columns]
        transform_df.columns = renamed_cols

        full_cols = list(self.reverse_mapping.keys())
        transform_df = self.pad_columns(transform_df, full_cols=full_cols)
        arr_result = self.scaler.inverse_transform(transform_df[full_cols])
        result_df = pd.DataFrame(arr_result, columns=full_cols, index=df.index)
        
        # Concat with non-numeric columns
        if non_numeric_cols:
            result_df = pd.concat([
                result_df.loc[:, renamed_cols],
                df[non_numeric_cols]
            ], axis=1)
        else:
            result_df = result_df.loc[:, renamed_cols]
            
        return self._input2orig_fmt(result_df[original_col_order])

    # --------------- Private helper methods ------------------
    def _instantiate_scaler(self, scaler_name: str, **kwargs):
        try:
            scaler_class = getattr(preprocessing, scaler_name)            
            scaler = scaler_class(**kwargs)
            return scaler
        except AttributeError:
            raise ValueError(f"Scaler '{scaler_name}' not found in sklearn.preprocessing")
        except TypeError as e:
            raise TypeError(f"Error instantiating {scaler_name}: {str(e)}")

    def _series2df(
        self,
        data: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """Convert input data to DataFrame format while preserving metadata."""
        if isinstance(data, pd.Series):
            self.is_series = True
            self.series_name = data.name
            return data.to_frame()
        self.is_series = False
        self.series_name = None
        return data
    
    def _input2orig_fmt(
        self,
        df: pd.DataFrame
    ) -> Union[pd.DataFrame, pd.Series]:
        """Convert DataFrame back to original format (Series or DataFrame)."""
        if self.is_series:
            if self.series_name is not None:
                df.columns = [self.series_name]
            return df.iloc[:, 0]
        return df

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify numeric columns in the DataFrame.
        """
        return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    def _store_params(self, df: pd.DataFrame):
        """
        Store all attributes that match the feature dimension.
        """
        n_features = df.shape[1]
        attributes = inspect.getmembers(self.scaler)
        
        for name, value in attributes:
            # Skip private attributes and callables
            if name.startswith('_') or callable(value):
                continue
                
            if isinstance(value, (np.ndarray, list)) and len(value) == n_features:
                for i, col in enumerate(df.columns):
                    if col not in self.reverse_mapping:
                        self.reverse_mapping[col] = {}
                    self.reverse_mapping[col][name] = value[i]
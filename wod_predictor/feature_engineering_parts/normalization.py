import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from ..constants import Constants as c

class BaseScaler(ABC):
    @abstractmethod
    def transform(self, df):
        pass

    @abstractmethod
    def reverse(self, df):
        pass

class PercentileScaler(BaseScaler):
    def __init__(self):
        self.reverse_mapping = None

    def reverse(self,ranked_data, reverse_mapping):
        # TODO: verify this works
        if self.reverse_mapping is None:
            raise ValueError("reverse_mapping is not set. Please run transform method first.")
        unranked_data = ranked_data.copy()
        for col in unranked_data.columns:
            index = (ranked_data[col]* len(reverse_mapping[col])).astype(int)-1
            unranked_data[col] = reverse_mapping[col][index].values
        return unranked_data

    def transform(self, df):
        # rank the data
        df = df.rank(pct=True)

        # create reverse mapping
        reverse_mapping = {}
        granularity = 0.001
        for col in df.columns:
            reverse_mapping[col] = df[col].quantile(np.arange(granularity)/(granularity-1)).values
        self.reverse_mapping = None

        return df

class StandardScalerByWod(BaseScaler):
    def __init__(self):
        self.scaler = StandardScaler()
        self.reverse_mapping = {}

    def transform(self, df):
        """
        Transform the data to have a mean of 0 and a standard deviation of 1.
        """
        columns = df.columns

        transformed_df = df.copy().astype(float)
        transformed_df.loc[:, columns] = self.scaler.fit_transform(df[columns])

        for i, col in enumerate(columns):
            self.reverse_mapping[col] = (self.scaler.mean_[i], self.scaler.scale_[i])

        return transformed_df

    def reverse(self, df):
        """
        Reverse the data transformation to its original scale. We may have a melted dataframe thats a series, so might be good to support that as well.
        """
        if self.reverse_mapping is None:
            raise ValueError("reverse_mapping is not set. Please run transform method first.")
        if isinstance(df, pd.Series):
            # TODO: implement reverse for melted dataframe
            raise NotImplementedError("reverse method for pd.Series is not implemented yet.")
        else:
            columns = df.columns
            for col in columns:
                col_clean = col.replace(c.workout_prefix, '')
                if col_clean not in self.reverse_mapping:
                    raise ValueError(f"Column {col_clean} is not in the reverse mapping.")
                mean, scale = self.reverse_mapping[col_clean]
                df[col] = df[col] * scale + mean
        
        return df
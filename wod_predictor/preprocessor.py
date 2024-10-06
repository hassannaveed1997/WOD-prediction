from functools import reduce
from .constants import Constants as c

import pandas as pd

from wod_predictor.feature_engineering_parts import (
    AthleteInfoFE,
    BenchmarkStatsFE,
    OpenResultsFE,
)


class DataPreprocessor:
    """
    Class for preprocessing data before modeling.

    Args:
        config (dict): Configuration parameters for the preprocessor.

    Methods:
        transform(data): Preprocesses the input data and returns the
        transformed data.

    """

    def __init__(self, config):
        self.config = config
        self.meta_data = {}
        self.open_fe_transformer = OpenResultsFE(**self.config.get("open_results", {}))
        self.benchmark_fe_transformer = BenchmarkStatsFE(
            **self.config.get("benchmark_stats", {})
        )
        self.athleteinfo_fe_transformer = AthleteInfoFE(**self.config.get("athlete_info", {}))

    def fit(self, data):
        # initalize all feature engineering objects
        self.open_fe_transformer.fit(
            data["open_results"], data.get("workout_descriptions", None)
        )

        if "benchmark_stats" in self.config:
            self.benchmark_fe_transformer.fit(data["benchmark_stats"])

        if "athlete_info" in self.config:
            self.athleteinfo_fe_transformer.fit(data["athlete_info"])

        return

    def transform(self, data):
        """
        Preprocesses the input data and returns the transformed data.

        This method transforms input data by calling
        corresponding methods and concatenating the results.

        Also one hot encodes categorical variables.
        
        Parameters:
            data (dict): Input data dictionary containing open results
            and benchmark stats.

        Returns:
            dict: Transformed data dictionary containing feature
            engineered data (X) and target variable (y).
        """

        fe_data = []

        X, y = self.transform_open_results(data)
        fe_data.append(X)

        if "benchmark_stats" in self.config:
            benchmark_stats = self.transform_benchmark_stats(data)
            fe_data.append(benchmark_stats)

        if "athlete_info" in self.config:
            athlete_info = self.transform_athlete_info(data)
            fe_data.append(athlete_info)

        # join all feature engineered data together
        fe_data = reduce(
            lambda left, right: pd.merge(left, right, on=c.athlete_id_col, how="left"),
            fe_data,
        )
        fe_data.drop(columns=[c.athlete_id_col], inplace=True)
        fe_data.index = X.index

        output = {"X": fe_data, "y": y, "meta_data": self.meta_data}
        return output

    def transform_open_results(self, data):
        """
        Transforms the open results data.

        Args:
            data (dict): Input data dictionary containing open results and workout descriptions.

        Returns:
            tuple: Transformed feature engineered data (X) and target variable (y).

        Raises:
            ValueError: If open results or workout descriptions are missing in the input data.

        """

        if "open_results" not in data or "workout_descriptions" not in data:
            raise ValueError(
                "Both open results and workout descriptions must be provided to transform open results"
            )

        open_results = self.open_fe_transformer.transform(
            data["open_results"], data.get("workout_descriptions", None)
        )
        y = open_results["score"]
        X = open_results.drop(columns=["score"])
        self.meta_data.update(self.open_fe_transformer.meta_data)

        return X, y

    def transform_benchmark_stats(self, data):
        """
        Transforms the benchmark stats data.

        Args:
            data (dict): Input data dictionary containing open results and benchmark stats.

        Returns:
            pandas.DataFrame: Transformed benchmark stats data.

        """

        # filter on intersecting athletes
        index = data["open_results"].index.intersection(
            data["benchmark_stats"].index
        )
        benchmark_df = data["benchmark_stats"].loc[index]

        benchmark_stats = self.benchmark_fe_transformer.transform(benchmark_df)
        return benchmark_stats

    def transform_athlete_info(self, data):
        """
        Transforms the athlete info data.

        Args:
            data (dict): Input data dictionary containing athlete info, and ope results.

        Returns:
            pandas.DataFrame: Transformed athlete info data.
        """

        # filter on intersecting athletes
        index = data["open_results"].index.intersection(
            data["athlete_info"].index
        )
        athlete_info_df = data["athlete_info"].loc[index]

        athlete_info = self.athleteinfo_fe_transformer.transform(athlete_info_df)

        return athlete_info

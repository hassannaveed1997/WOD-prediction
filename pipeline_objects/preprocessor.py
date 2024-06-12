from pipeline_objects.feature_engineering_parts import (
    OpenResultsFE,
    BenchmarkStatsFE,
)
from functools import reduce
import pandas as pd


class DataPreprocessor:
    """
    Class for preprocessing data before modeling.

    Args:
        config (dict): Configuration parameters for the preprocessor.

    Methods:
        transform(data): Preprocesses the input data and returns the transformed data.

    """

    def __init__(self, config):
        self.config = config

    def transform(self, data):
        """
                Preprocesses the input data and returns the transformed data.
        `
                Args:
                    data (dict): Input data dictionary containing open results and benchmark stats.

                Returns:
                    dict: Transformed data dictionary containing feature engineered data (X) and target variable (y).

                Raises:
                    ValueError: If both open results and workout descriptions are not provided.
                    NotImplementedError: If athlete info transformation is not yet implemented.

        """

        fe_data = []

        X, y = self.transform_open_results(data)
        fe_data.append(X)

        if "benchmark_stats" in self.config:
            benchmark_stats = self.transform_benchmark_stats(data)
            fe_data.append(benchmark_stats)

        if "athlete_info" in self.config:
            raise NotImplementedError(
                "Athlete info transformation not yet implemented"
            )

        # join all feature engineered data together
        fe_data = reduce(
            lambda left, right: pd.merge(left, right, how="left"), fe_data
        )

        # one hot encode categorical variables
        for col in fe_data.columns:
            if fe_data[col].dtype == "object":
                fe_data = pd.concat(
                    [fe_data, pd.get_dummies(fe_data[col], prefix=col)], axis=1
                )
                fe_data.drop(col, axis=1, inplace=True)

        output = {"X": fe_data, "y": y}
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

        open_results_fe = OpenResultsFE(**self.config["open_results"])
        open_results = open_results_fe.transform(
            data["open_results"], data.get("workout_descriptions", None)
        )
        y = open_results["score"]
        X = open_results.drop(columns=["score"])

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

        benchmark_stats_fe = BenchmarkStatsFE(**self.config["benchmark_stats"])
        benchmark_stats = benchmark_stats_fe.transform(benchmark_df)
        return benchmark_stats

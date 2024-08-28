from wod_predictor.feature_engineering_parts import OpenResultsFE, BenchmarkStatsFE
from functools import reduce
from .constants import Constants as c
import pandas as pd


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.meta_data = {}
        self.open_fe_transformer = OpenResultsFE(**self.config.get("open_results", {}))
        self.benchmark_fe_transformer = BenchmarkStatsFE(
            **self.config.get("benchmark_stats", {})
        )

    def fit(self, data):
        # initalize all feature engineering objects
        self.open_fe_transformer.fit(
            data["open_results"], data.get("workout_descriptions", None)
        )

        if "benchmark_stats" in self.config:
            self.benchmark_fe_transformer.fit(data["benchmark_stats"])

        if "athlete_info" in self.config:
            raise NotImplementedError("Athlete info transformation not yet implemented")

        # TODO: maintain a list of columns in the resulting dataframe (for test_df) to ensure that the columns are the same

        return

    def transform(self, data):
        fe_data = []

        X, y = self.transform_open_results(data)
        fe_data.append(X)

        if "benchmark_stats" in self.config:
            benchmark_stats = self.transform_benchmark_stats(data)
            fe_data.append(benchmark_stats)

        if "athlete_info" in self.config:
            raise NotImplementedError("Athlete info transformation not yet implemented")

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
        # filter on intersecting athletes
        index = data["open_results"].index.intersection(data["benchmark_stats"].index)
        benchmark_df = data["benchmark_stats"].loc[index]

        benchmark_stats = self.benchmark_fe_transformer.transform(benchmark_df)
        return benchmark_stats

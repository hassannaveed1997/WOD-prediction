from wod_predictor.feature_engineering_parts import OpenResultsFE, BenchmarkStatsFE
from functools import reduce
from .constants import Constants as c
import pandas as pd


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.meta_data = {}

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
        fe_data = reduce(lambda left, right: pd.merge(left, right, on = c.athlete_id_col, how="left"), fe_data)
        fe_data.drop(columns=[c.athlete_id_col], inplace=True)
        fe_data.index = X.index

        # one hot encode categorical variables
        for col in fe_data.columns:
            if fe_data[col].dtype == "object":
                fe_data = pd.concat(
                    [fe_data, pd.get_dummies(fe_data[col], prefix=col)], axis=1
                )
                fe_data.drop(col, axis=1, inplace=True)


        output = {"X": fe_data, "y": y, 'meta_data': self.meta_data}
        return output

    def transform_open_results(self, data):
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
        self.meta_data.update(open_results_fe.meta_data)

        return X, y

    def transform_benchmark_stats(self, data):
        # filter on intersecting athletes
        index = data["open_results"].index.intersection(data["benchmark_stats"].index)
        benchmark_df = data["benchmark_stats"].loc[index]

        benchmark_stats_fe = BenchmarkStatsFE(**self.config["benchmark_stats"])
        benchmark_stats = benchmark_stats_fe.transform(benchmark_df)
        return benchmark_stats

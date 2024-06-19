from pipeline_objects.feature_engineering_parts import OpenResultsFE, BenchmarkStatsFE, generate_meta_data
from functools import reduce
import pandas as pd


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

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
        fe_data = reduce(lambda left, right: pd.merge(left, right, how="left"), fe_data)
        fe_data.index = X.index
        
        # one hot encode categorical variables
        for col in fe_data.columns:
            if fe_data[col].dtype == "object":
                fe_data = pd.concat(
                    [fe_data, pd.get_dummies(fe_data[col], prefix=col)], axis=1
                )
                fe_data.drop(col, axis=1, inplace=True)

        meta_data = generate_meta_data(fe_data)

        output = {"X": fe_data, "y": y, 'meta_data': meta_data}
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

        return X, y

    def transform_benchmark_stats(self, data):
        # filter on intersecting athletes
        index = data["open_results"].index.intersection(data["benchmark_stats"].index)
        benchmark_df = data["benchmark_stats"].loc[index]

        benchmark_stats_fe = BenchmarkStatsFE(**self.config["benchmark_stats"])
        benchmark_stats = benchmark_stats_fe.transform(benchmark_df)
        return benchmark_stats

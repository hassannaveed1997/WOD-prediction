from .base import BaseFEPipelineObject
from .helpers import remove_outliers, fill_missing_values


class BenchmarkStatsFE(BaseFEPipelineObject):
    def __init__(self, remove_outliers: bool = True, missing_method="knn", **kwargs):
        self.remove_outliers = remove_outliers
        self.missing_method = missing_method
        self.kwargs = kwargs

    def transform(self, benchmark_data):
        """
        This function is intended to perform a few operations:
        - fill missing values
        - remove outliers
        """
        # remove unnecessary columns
        if "name" in benchmark_data.columns:
            benchmark_data.drop(columns=["name"], inplace=True)

        # remove outliers
        if self.remove_outliers:
            benchmark_data = remove_outliers(benchmark_data, **self.kwargs)

        # fill missing values
        if self.missing_method is not None:
            benchmark_data = fill_missing_values(
                benchmark_data, method=self.missing_method, **self.kwargs
            )

        # keep athlete_id as index
        benchmark_data['athlete_id'] = benchmark_data.index

        return benchmark_data

from .base import BaseFEPipelineObject
from .helpers import fill_missing_values
from .outlier_detection import IQRoutlierDetector
from ..constants import Constants as c


class BenchmarkStatsFE(BaseFEPipelineObject):
    def __init__(self, remove_outliers: bool = True, missing_method="knn", **kwargs):
        if remove_outliers:
            self.outlier_remover = IQRoutlierDetector()
        else:
            self.outlier_remover = None
        self.missing_method = missing_method
        self.kwargs = kwargs

    def fit(self, benchmark_data):
        """
        Initialize any transformers for later use in transform
        """
        if self.outlier_remover:
            self.outlier_remover.fit(benchmark_data)
        return

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
        if self.outlier_remover:
            benchmark_data = self.outlier_remover.transform(
                benchmark_data, **self.kwargs
            )

        # fill missing values
        if self.missing_method is not None:
            benchmark_data = fill_missing_values(
                benchmark_data, method=self.missing_method, **self.kwargs
            )

        # keep athlete_id as index
        benchmark_data[c.athlete_id_col] = benchmark_data.index

        return benchmark_data

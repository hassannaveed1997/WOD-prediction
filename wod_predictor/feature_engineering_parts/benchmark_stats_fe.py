import warnings
from typing import Literal, Optional, Dict, Any

from .base import BaseFEPipelineObject
from .helpers import fill_missing_values
from .outlier_detection import IQRoutlierDetector
from ..constants import Constants as c
from .misc import DropFeatures
from .normalization import QuantileScaler, StandardScalerByWod, GenericSklearnScaler

class BenchmarkStatsFE(BaseFEPipelineObject):
    """
    Feature engineering pipeline object for benchmark statistics.

    Args:
        remove_outliers (bool, optional): Flag indicating whether to remove outliers. Defaults to True.
        missing_method (str, optional): Method for filling missing values. Defaults to "knn".
        **kwargs: Additional keyword arguments to be passed.

    Methods:
        transform(benchmark_data): Performs operations on the benchmark data.

    """

    def __init__(
        self,
        remove_outliers: bool = True,
        missing_method: str = "knn",
        drop_missing_threshold: float = None,
        scale_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.missing_method = missing_method
        self.kwargs = kwargs
        
        if remove_outliers:
            self.outlier_remover = IQRoutlierDetector()
        else:
            self.outlier_remover = None

        if drop_missing_threshold:
            self.drop_features = DropFeatures(drop_missing_threshold)
        else:
            self.drop_features = None
        
        if scale_args is not None:
            scale_method = scale_args.pop("method")
            if scale_method == "standard":
                self.scaler = StandardScalerByWod()
            elif scale_method == "quantile":
                self.scaler = QuantileScaler()
            elif scale_method == "general":
                assert "scaler_name" in scale_args, "To use general scaler, you must specify scaler_name"
                self.scaler = GenericSklearnScaler(**scale_args)
            else:
                warnings.warn(f"{scale_method} is currently not implemented. No scaling is performed for benchmark_stats")
                self.scaler = None

    def fit(self, benchmark_data):
        """
        Initialize any transformers for later use in transform
        """
        df_copy = benchmark_data.copy()
        if self.outlier_remover:
            self.outlier_remover.fit(benchmark_data)
            df_copy = self.outlier_remover.transform(df_copy)
        if self.drop_features:
            self.drop_features.fit(benchmark_data)
            df_copy = self.drop_features.transform(df_copy)
        if self.missing_method is not None:
            df_copy = fill_missing_values(
                df_copy, method=self.missing_method, **self.kwargs
            )

        if self.scaler:
            self.scaler.fit(df_copy.drop(columns=["name"]))
        return

    def transform(self, benchmark_data):
        """
        This function is intended to perform a few operations:
        - fill missing values
        - remove outliers

        Args:
            benchmark_data (DataFrame): The benchmark data to be transformed.

        Returns:
            DataFrame: The transformed benchmark data.

        """
        # remove unnecessary columns
        if "name" in benchmark_data.columns:
            benchmark_data.drop(columns=["name"], inplace=True)

        # remove outliers
        if self.outlier_remover:
            benchmark_data = self.outlier_remover.transform(
                benchmark_data, **self.kwargs
            )
        
        # drop features with more than threshold percentage of missing values
        if self.drop_features:
            benchmark_data = self.drop_features.transform(benchmark_data)

        # fill missing values
        if self.missing_method is not None:
            benchmark_data = fill_missing_values(
                benchmark_data, method=self.missing_method, **self.kwargs
            )

        # scale the data
        if self.scaler:
            # TODO: we refit the scaler here, which is not ideal.
            # We should fit the scaler in the fit method,
            # but due to change of distribution after filling missing values,
            # and removing outliers, we must refit the scaler here.
            # This should be resolved in a fuure PR.
            benchmark_data = self.scaler.transform(benchmark_data)

        # keep athlete_id as index
        benchmark_data[c.athlete_id_col] = benchmark_data.index
        return benchmark_data

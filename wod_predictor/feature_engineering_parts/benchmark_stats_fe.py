import warnings
from typing import Any, Dict, Literal, Optional

from wod_predictor.feature_engineering_parts.base import TransformerMixIn

from ..constants import Constants as c
from .utils.misc import DropFeatures
from .utils.missing_values import MissingValueImputation
from .utils.normalization import (GenericSklearnScaler, QuantileScaler,
                                  StandardScalerByWod)
from .utils.outlier_detection import IQRoutlierDetector


class BenchmarkStatsFE(TransformerMixIn):
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
        **kwargs,
    ):
        self.missing_method = missing_method
        self.kwargs = kwargs
        self.transformers = []
        if remove_outliers:
            self.transformers.append(IQRoutlierDetector())

        if drop_missing_threshold:
            self.transformers.append(DropFeatures(drop_missing_threshold))

        if missing_method is not None:
            self.transformers.append(
                MissingValueImputation(method=missing_method, **kwargs)
            )
        self._initialize_scaler(scale_args)
        if self.scaler:
            self.transformers.append(self.scaler)

    def _initialize_scaler(self, scale_args):
        if scale_args is not None:
            scale_args_ = scale_args.copy()
            scale_method = scale_args_.pop("method")
            if scale_method == "standard":
                self.scaler = StandardScalerByWod()
            elif scale_method == "quantile":
                self.scaler = QuantileScaler()
            elif scale_method == "general":
                assert (
                    "scaler_name" in scale_args
                ), "To use general scaler, you must specify scaler_name"
                self.scaler = GenericSklearnScaler(**scale_args_)
            else:
                warnings.warn(
                    f"{scale_method} is currently not implemented. No scaling is performed for benchmark_stats"
                )
                self.scaler = None
        else:
            self.scaler = None

    def transform(self, benchmark_data):
        """
        This function is intended to run all the transformers initialized earlier

        Args:
            benchmark_data (DataFrame): The benchmark data to be transformed.

        Returns:
            DataFrame: The transformed benchmark data.

        """
        # remove unnecessary columns
        if "name" in benchmark_data.columns:
            benchmark_data.drop(columns=["name"], inplace=True)

        # run all transformation pipelines
        for transformer in self.transformers:
            benchmark_data = transformer.transform(benchmark_data)

        # keep athlete_id as index
        benchmark_data[c.athlete_id_col] = benchmark_data.index
        return benchmark_data

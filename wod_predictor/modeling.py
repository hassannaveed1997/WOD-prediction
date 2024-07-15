from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .models.helpers import show_breakdown_by_workout, unstack_series
import pandas as pd
import numpy as np


class BaseModeler:
    def __init__(self, config: dict = {}):
        self.config = config
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def show_results(self, y_test, meta_data={}):
        if self.y_pred is None:
            raise ValueError("No predictions to show, plase train model")

        self.y_test = y_test
        # show mean absolute error
        print(
            "Mean Absolute Error:",
            round(mean_absolute_error(self.y_test, self.y_pred), 2),
        )
        print(
            "Mean Absolute Percentage Error:",
            round(mean_absolute_percentage_error(self.y_test, self.y_pred), 2),
        )

        if "idx_to_workout_name" in meta_data and "idx_to_athlete_id" in meta_data:
            y_test_unstacked = unstack_series(self.y_test, meta_data)
            y_pred_unstacked = unstack_series(
                pd.Series(self.y_pred, index=self.y_test.index, name="score"), meta_data
            )

            # reverse the scaling
            if "scaler" in meta_data:
                y_test_unstacked = meta_data["scaler"].reverse(y_test_unstacked)
                y_pred_unstacked = meta_data["scaler"].reverse(y_pred_unstacked)

            show_breakdown_by_workout(y_pred_unstacked, y_test_unstacked)


class RandomForestModel(BaseModeler):
    def __init__(self, **kwargs):
        config = kwargs.pop("config", None)
        self.kwargs = kwargs
        super().__init__(config=config)

    def fit(self, X, y):
        # split data
        self.x_train = X
        self.y_train = y

        # fit model
        self.model = RandomForestRegressor(**self.kwargs)
        self.model.fit(self.x_train, self.y_train)

    def predict(self, X):
        self.x_test = X
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def show_results(self, **kwargs):
        super().show_results(**kwargs)

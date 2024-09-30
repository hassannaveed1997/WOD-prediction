from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .models.helpers import show_breakdown_by_workout
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

    def show_results(self, y_test, meta_data = {}):
        if self.y_pred is None:
            raise ValueError("No predictions to show, plase train model")
        
        self.y_test = y_test
        # show mean absolute error
        print(
            "Mean Absolute Error:", round(mean_absolute_error(self.y_test, self.y_pred), 2)
        )
        print(
            "Mean Absolute Percentage Error:",
            round(mean_absolute_percentage_error(self.y_test, self.y_pred), 2),
        )
        if 'idx_to_workout_name' in meta_data:
            show_breakdown_by_workout(y_test = self.y_test, y_pred = self.y_pred, mapping = meta_data['idx_to_workout_name'])

    
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


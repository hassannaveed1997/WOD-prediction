from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .models.helpers import show_breakdown_by_workout
import pandas as pd
import numpy as np


class BaseModeler:
    def __init__(self, meta_data: dict = {}, config: dict = {}):
        self.meta_data = meta_data
        self.config = config
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def show_results(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        if self.x_test is None or self.y_test is None:
            raise ValueError("Data has not been split yet")

        y_pred = self.model.predict(self.x_test)
        # show mean absolute error
        print(
            "Mean Absolute Error:", round(mean_absolute_error(self.y_test, y_pred), 2)
        )
        print(
            "Mean Absolute Percentage Error:",
            round(mean_absolute_percentage_error(self.y_test, y_pred), 2),
        )

        if 'idx_to_workout_name' in self.meta_data and 'idx_to_athlete_id' in self.meta_data:
            y_test_unstacked = self.unstack_series(self.y_test)
            y_pred_unstacked = self.unstack_series(pd.Series(y_pred, index = self.y_test.index, name='score'))

            # reverse the scaling
            if 'scaler' in self.meta_data:
                y_test_unstacked = self.meta_data['scaler'].reverse(y_test_unstacked)
                y_pred_unstacked = self.meta_data['scaler'].reverse(y_pred_unstacked)

            show_breakdown_by_workout(y_pred_unstacked, y_test_unstacked)

    def unstack_series(self, series):
        """
        Unstack a series with a multiindex
        """
        df = pd.DataFrame(series)
        df['workout_name'] = df.index.map(self.meta_data['idx_to_workout_name'])
        df['athlete_id'] = df.index.map(self.meta_data['idx_to_athlete_id'])
        return df.pivot(columns='workout_name', values='score', index = 'athlete_id')

    def split_data(self, X, y, method="random"):
        test_size = self.config.get("test_size", 0.2)
        test_index = X.index

        # filter test set
        if "test_filter" in self.config:
            if 'idx_to_workout_name' not in self.meta_data:
                raise KeyError("meta_data must have idx_to_workout_name mapping to use test_filter. Either pass in mapping or set test_filter to None")
            mapping = self.meta_data['idx_to_workout_name']
            useable_indices = mapping[mapping.str.contains(self.config['test_filter'], regex = True)].index
            test_index = test_index.intersection(useable_indices)

        if method == "random":
            # sample from the test flag with value 1
            test_index = np.random.choice(test_index, int(len(test_index) * test_size), replace=False)

        self.x_train = X.drop(test_index)
        self.y_train = y.drop(test_index)
        self.x_test = X.loc[test_index]
        self.y_test = y.loc[test_index]
    
class RandomForestModel(BaseModeler):
    def __init__(self, **kwargs):
        meta_data = kwargs.pop("meta_data", None)
        config = kwargs.pop("config", None)
        self.kwargs = kwargs
        super().__init__(meta_data=meta_data, config=config)

    def fit(self, X, y):
        # split data
        self.split_data(X, y)

        # fit model
        self.model = RandomForestRegressor(**self.kwargs)
        self.model.fit(self.x_train, self.y_train)

    def show_results(self, **kwargs):
        super().show_results(**kwargs)


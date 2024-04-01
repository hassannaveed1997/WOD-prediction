from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

class BaseModeler:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError
    
    def show_results(self):
        if self.model is None:
            raise ValueError('Model has not been trained yet')
        if self.x_test is None or self.y_test is None:
            raise ValueError('Data has not been split yet')
        
        y_pred = self.model.predict(self.x_test)
        # show mean absolute error
        print('Mean Absolute Error:', round(mean_absolute_error(self.y_test, y_pred), 2))
    
    def split_data(self, X, y, method = 'random'):
        if method == 'random':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)


class RandomForestModel(BaseModeler):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def fit(self, X, y):
        # one hot encode categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                X = pd.concat([X, pd.get_dummies(X[col], prefix=col)], axis=1)
                X.drop(col, axis=1, inplace=True)

        # split data
        self.split_data(X, y)

        # fit model
        self.model = RandomForestRegressor(**self.kwargs)
        self.model.fit(self.x_train, self.y_train)

    def show_results(self):
        super().show_results()

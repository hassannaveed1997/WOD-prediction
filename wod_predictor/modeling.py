from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .models.helpers import show_breakdown_by_workout, unstack_series
import pandas as pd
import numpy as np
import torch
from torch import nn





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
        print('random forest method called')
        self.x_test = X
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def show_results(self, **kwargs):
        super().show_results(**kwargs)

class NeuralNetV0(BaseModeler,nn.Module):
    def __init__(self, input_features, hidden_units, output_features, config: dict = {}):
        BaseModeler.__init__(self)
        nn.Module.__init__(self)

        # Define the layers in the model
        self.model = nn.Sequential(
        nn.Linear(in_features = input_features, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        #nn.Dropout(p=0.2),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = output_features),
        )
    
    def forward(self, x):
        return self.model(x)

    def fit(self, X, y, epochs=1000, lr=0.001):

        X_train_torch = torch.tensor(X.values,dtype = torch.float32)
        y_train_torch = torch.tensor(y.values,dtype = torch.float32).view(-1,1)

        # Define the loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()  # Set the model in training mode

            # Forward pass
            y_pred = self.forward(X_train_torch)
            loss = loss_fn(y_pred, y_train_torch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} | Training Loss: {loss.item():.4f}")

    def predict(self, X_test):

        X_test_torch = torch.tensor(X_test.values,dtype = torch.float32)

        print("Predict method called")
        self.eval()  # Set the model in evaluation mode
        with torch.inference_mode():
            self.y_pred = self.forward(X_test_torch).detach().numpy().squeeze()  # Convert predictions to numpy
        print(self.y_pred.shape)
        return self.y_pred

    def show_results(self, **kwargs):
        super().show_results(**kwargs)
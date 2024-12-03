from time import time
from copy import deepcopy
from sklearn.metrics import mean_absolute_error

from wod_predictor.hyperparam_search import helpers


def model_only_objective(config):
    model_params = deepcopy(config['model'])
    data = deepcopy(config['data'])
    train_data = data['train_data']
    val_data = data['val_data']

    model_name = model_params['name']
    init_args = model_params['init_args']
    fit_args = model_params['fit_args']
    
    model = helpers.fit_model(
        model_name,
        train_data['X'],
        train_data['y'],
        init_args,
        fit_args
    )

    y_pred = model.predict(val_data['X'])
    score = mean_absolute_error(val_data['y'], y_pred)    
    return score
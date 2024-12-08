import warnings, logging, argparse
import pickle, yaml

import wod_predictor
from wod_predictor.data_loader import DataLoader
from wod_predictor.splitter import DataSplitter
from wod_predictor.preprocessor import DataPreprocessor 
from wod_predictor.modeling import RandomForestModel
from wod_predictor.hyperparam_search.tuner import HyperparamTuner, ParamRange
from wod_predictor.hyperparam_search import helpers
from wod_predictor.hyperparam_search.objectives import model_only_objective

from copy import deepcopy
from sklearn.metrics import mean_absolute_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    return parser.parse_args()

def start_study(config, objective_fn):
    param_ranges = {}
    for name, args in config['tune_params'].items():
        param_ranges[name] = ParamRange(**args)
    
    tuner = HyperparamTuner(
        objective_fn=objective_fn,
        param_ranges=param_ranges,
        base_config=config,
        n_trials=config['study']['tuner_args']['n_trials']
    )
    final_config = tuner.optimize()
    return tuner, final_config

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    train, val, test = helpers.load_train_val_test_data()
    preprocessor = helpers.fit_preprocessor(
        train, preprocessor_args=config["preprocessing"]
    )
    train_processed = preprocessor.transform(train)
    val_processed = preprocessor.transform(val)
    test_processed = preprocessor.transform(test)

    config['data'] = {
        "train_data": train_processed,
        "val_data": val_processed
    }

    # Default model
    model_name = config['model']['name']
    model = helpers.fit_model(
        model_name,
        train_processed['X'], train_processed['y'],
        init_args=config['model']['init_args'],
        fit_args=config['model']['fit_args']
    )
    val_preds = model.predict(val_processed['X'])
    val_mae = mean_absolute_error(val_preds, val_processed['y'])
    logging.info(f'Default model val MAE: {val_mae}')
    test_preds = model.predict(test_processed['X'])
    test_mae = mean_absolute_error(test_preds, test_processed['y'])
    logging.info(f'Default model test MAE: {test_mae}')

    tuner, final_config = start_study(config, model_only_objective)
    model = helpers.fit_model(
        model_name,
        train_processed['X'], train_processed['y'],
        init_args=final_config['model']['init_args'],
        fit_args=final_config['model']['fit_args']
    )
    val_preds = model.predict(val_processed['X'])
    val_mae = mean_absolute_error(val_preds, val_processed['y'])
    logging.info(f'Best model val MAE: {val_mae}')
    test_preds = model.predict(test_processed['X'])
    test_mae = mean_absolute_error(test_preds, test_processed['y'])
    logging.info(f'Best model test MAE: {test_mae}')
    tuner.save_best_state('modeling/kenneth/results/random_forest')
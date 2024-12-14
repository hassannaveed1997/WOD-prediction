import logging, pickle, os
from copy import deepcopy
from typing import Dict, List, Union, Optional, Callable, Any
from dataclasses import dataclass

import optuna
import numpy as np
from sklearn.exceptions import NotFittedError

@dataclass
class ParamRange:
    """Defines the range for a hyperparameter"""
    param_type: str  # 'int', 'float', 'categorical'
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List] = None
    log: bool = False
    path: Optional[List[str]] = None  # Path to nested parameter
    
    def __post_init__(self):
        if self.path is None:
            self.path = []

class HyperparamTuner:
    def __init__(
        self,
        objective_fn: Callable,
        param_ranges: Dict[str, ParamRange],
        base_config: Dict[str, Any],
        n_trials: int = 100,
        direction: str = "minimize",
        seed: int = 42,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            objective_fn: Callable function that takes trial parameters and returns a score
            param_ranges: Dictionary of parameter ranges to optimize
            base_config: Base configuration dictionary that will be updated with trial params
            n_trials: Number of optimization trials
            direction: Optimization direction ('minimize' or 'maximize')
            seed: Random seed
        """
        self.objective_fn = objective_fn
        self.param_ranges = param_ranges
        self.base_config = base_config
        self.n_trials = n_trials
        self.direction = direction
        self.seed = seed
        
        self.study = None
        self.best_params = None
        self.best_score = None
        self.final_config = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def optimize(self):
        """Run the hyperparameter optimization process."""
        self.logger.info(f"Start hyperparameter optimization for {self.n_trials} trials")
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        self.study.optimize(
            self._create_objective(),
            n_trials=self.n_trials,
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Create final config with best parameters
        final_config = deepcopy(self.base_config)
        for param_name, param_range in self.param_ranges.items():
            value = self.best_params[param_name]
            if param_range.path:
                final_config = self._update_nested_dict(final_config, param_range.path, value)
            else:
                final_config[param_name] = value
        final_config.pop('data')
        self.logger.info(f"Best score: {self.best_score}")
        self.final_config = final_config
        return final_config
    
    # ---------- Private methods ----------
    def _update_nested_dict(self, d: dict, path: List[str], value: Any) -> dict:
        """Update a nested dictionary at the specified path with the given value."""
        current = d
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value
        return d
    
    def _create_objective(self):
        def objective(trial):
            config = deepcopy(self.base_config)
            
            # Get parameters from trial and update config
            for param_name, param_range in self.param_ranges.items():
                if param_range.param_type == "int":
                    value = trial.suggest_int(
                        param_name, param_range.low, param_range.high, log=param_range.log
                    )
                elif param_range.param_type == "float":
                    value = trial.suggest_float(
                        param_name, param_range.low, param_range.high, log=param_range.log
                    )
                elif param_range.param_type == "categorical":
                    value = trial.suggest_categorical(
                        param_name, param_range.choices
                    )
                else:
                    raise ValueError(f"Invalid param type: {param_range.param_type}")
                
                # Update the config at the specified path
                if param_range.path:
                    config = self._update_nested_dict(config, param_range.path, value)
                else:
                    config[param_name] = value
            
            # Call user's objective function with the updated config
            return self.objective_fn(config)
        
        return objective

    def save_best_state(self, save_folder: str):
        path = os.path.join(save_folder, "best_state.pkl")
        os.makedirs(save_folder, exist_ok=True)
        if self.final_config is None:
            raise NotFittedError("No study has been created yet.")
        with open(path, 'wb') as handle:
            pickle.dump(self.final_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
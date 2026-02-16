"""
Shared hyperparameter tuning utilities.

Extracted from tests/experiments/hyperparameter_tuning/lstm_hyperparameter_tuning.py
for use by both the experimental tuning scripts and the production finetune API.
"""

import numpy as np
from typing import Dict, List, Optional, Any


def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif obj is None:
        return None
    else:
        return obj


class HyperparameterConfig:
    """Configuration for a single hyperparameter to tune"""

    def __init__(self, name: str, min_val: float, max_val: float,
                 step: Optional[float] = None, type: str = 'int',
                 scale: str = 'linear', values: Optional[List] = None):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.type = type
        self.scale = scale
        self.values = values

    def get_grid_values(self) -> List:
        """Get all values for grid search"""
        if self.values is not None:
            return self.values

        if self.type == 'int':
            step = self.step or 1
            return list(range(int(self.min_val), int(self.max_val) + 1, int(step)))
        else:
            step = self.step or (self.max_val - self.min_val) / 10
            values = []
            val = self.min_val
            while val <= self.max_val:
                values.append(val)
                val += step
            return values

    def sample_random(self) -> Any:
        """Sample a random value"""
        if self.values is not None:
            return np.random.choice(self.values)

        if self.scale == 'log':
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            value = 10 ** np.random.uniform(log_min, log_max)
        else:
            value = np.random.uniform(self.min_val, self.max_val)

        if self.type == 'int':
            return int(round(value))
        else:
            return float(value)


class RandomSearch:
    """Random sampling from parameter space"""

    def __init__(self, tunable_params: Dict[str, HyperparameterConfig],
                 n_iter: int = 50):
        self.tunable_params = tunable_params
        self.n_iter = n_iter

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for _ in range(self.n_iter):
            config = {}
            for name, param in self.tunable_params.items():
                config[name] = param.sample_random()
            yield config

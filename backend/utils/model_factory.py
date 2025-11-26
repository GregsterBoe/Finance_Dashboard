"""
Shared model factory for creating ML model instances.
Centralizes model creation logic to avoid duplication.
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from models.ml_models import ModelConfig, ModelType
from models.lstm_model import LSTMStockPredictor


def create_model(config: ModelConfig, model_dir: str = 'saved_models/production'):
    """
    Create a model instance based on configuration.

    Args:
        config: Model configuration specifying type and parameters
        model_dir: Directory for saving LSTM models (default: 'saved_models/production')

    Returns:
        Model instance ready for training

    Raises:
        ValueError: If model type is not supported
    """
    if config.model_type == ModelType.DECISION_TREE:
        return DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state
        )
    elif config.model_type == ModelType.RANDOM_FOREST:
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state
        )
    elif config.model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegression()
    elif config.model_type == ModelType.LSTM:
        return LSTMStockPredictor(
            sequence_length=config.sequence_length,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            model_dir=model_dir
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

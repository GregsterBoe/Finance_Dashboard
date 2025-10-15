"""
backend/services/metrics_service.py

Centralized metrics calculation service for ML models.
Single source of truth for all performance metrics.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.ml_models import TrainingMetrics


class MetricsService:
    """Service for calculating standardized ML performance metrics"""
    
    @staticmethod
    def calculate_regression_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        training_samples: int = 0
    ) -> TrainingMetrics:
        """
        Calculate standard regression metrics for price prediction.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            training_samples: Number of samples used for training
            
        Returns:
            TrainingMetrics object with all calculated metrics
        """
        # Ensure arrays are 1D and same length
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()
        
        if len(actual) != len(predicted):
            raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
        
        if len(actual) == 0:
            raise ValueError("Cannot calculate metrics on empty arrays")
        
        # Calculate errors
        errors = predicted - actual
        
        # Core metrics
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        r2 = float(r2_score(actual, predicted))
        
        # MAPE (with safety for zero values)
        mape = float(np.mean(np.abs(errors / (np.abs(actual) + 1e-8))) * 100)
        
        # Directional accuracy (if we have sequential data)
        directional_accuracy = None
        if len(actual) > 1:
            directional_accuracy = MetricsService._calculate_directional_accuracy(
                actual, predicted
            )
        
        return TrainingMetrics(
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r2_score=round(r2, 4),
            mape=round(mape, 2),
            training_samples=training_samples,
            directional_accuracy=round(directional_accuracy, 2) if directional_accuracy else None
        )
    
    @staticmethod
    def _calculate_directional_accuracy(
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (percentage of correct up/down predictions).
        
        Args:
            actual: Array of actual values (time-ordered)
            predicted: Array of predicted values (time-ordered)
            
        Returns:
            Directional accuracy as percentage (0-100)
        """
        if len(actual) < 2:
            return 0.0
        
        # Calculate direction changes
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        
        # Count correct predictions
        correct = np.sum(actual_direction == predicted_direction)
        total = len(actual_direction)
        
        return float((correct / total) * 100)
    
    @staticmethod
    def calculate_from_predictions(
        predictions: List,
        actual_key: str = "actual_close",
        predicted_key: str = "predicted_close",
        training_samples_key: str = "training_samples"
    ) -> TrainingMetrics:
        """
        Calculate metrics from a list of prediction objects/dicts.
        
        Args:
            predictions: List of predictions (dicts or objects with actual/predicted values)
            actual_key: Key/attribute name for actual values
            predicted_key: Key/attribute name for predicted values
            training_samples_key: Key/attribute name for training samples count
            
        Returns:
            TrainingMetrics object
        """
        if not predictions:
            raise ValueError("Cannot calculate metrics from empty predictions list")
        
        # Extract values (handle both dict and object attributes)
        def get_value(obj, key):
            return obj[key] if isinstance(obj, dict) else getattr(obj, key)
        
        actual = np.array([get_value(p, actual_key) for p in predictions])
        predicted = np.array([get_value(p, predicted_key) for p in predictions])
        
        # Get training samples from first prediction (should be same for all)
        training_samples = get_value(predictions[0], training_samples_key)
        
        return MetricsService.calculate_regression_metrics(
            actual=actual,
            predicted=predicted,
            training_samples=training_samples
        )
    
    @staticmethod
    def calculate_from_model_output(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        training_samples: int
    ) -> TrainingMetrics:
        """
        Calculate metrics directly from model training/validation output.
        Useful for LSTM and other models that return raw predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            training_samples: Number of training samples
            
        Returns:
            TrainingMetrics object
        """
        return MetricsService.calculate_regression_metrics(
            actual=y_true,
            predicted=y_pred,
            training_samples=training_samples
        )
    
    @staticmethod
    def format_metric_value(metric_name: str, value: float) -> str:
        """
        Format metric value for display with appropriate precision.
        
        Args:
            metric_name: Name of the metric (rmse, mae, etc.)
            value: Metric value
            
        Returns:
            Formatted string
        """
        if metric_name in ['mape', 'directional_accuracy']:
            return f"{value:.2f}%"
        elif metric_name in ['rmse', 'mae', 'r2_score']:
            return f"{value:.4f}"
        elif metric_name == 'training_samples':
            return f"{int(value):,}"
        else:
            return f"{value:.4f}"
    
    @staticmethod
    def get_metric_display_name(metric_name: str) -> str:
        """Get human-readable display name for a metric"""
        display_names = {
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE',
            'r2_score': 'R² Score',
            'directional_accuracy': 'Directional Accuracy',
            'training_samples': 'Training Samples'
        }
        return display_names.get(metric_name, metric_name.replace('_', ' ').title())


# Singleton instance
_metrics_service = None

def get_metrics_service() -> MetricsService:
    """Get the singleton metrics service instance"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
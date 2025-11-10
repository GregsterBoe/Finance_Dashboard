"""
Simple Confidence-Based Abstention using Monte Carlo Dropout

This is a SIMPLER alternative to the dual-head selective LSTM.
Instead of training a confidence head, we use the existing LSTM
and estimate uncertainty using Monte Carlo Dropout.

Pros:
- No architecture changes needed
- No special loss function
- Works with any trained LSTM
- Easy to tune threshold

Cons:
- Not learned (heuristic-based)
- Requires multiple forward passes
- May be less optimal than learned abstention
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMStockPredictor


class ConfidenceEstimator:
    """
    Wrapper around standard LSTM that adds confidence estimation
    using Monte Carlo Dropout
    """
    
    def __init__(self, lstm_model: LSTMStockPredictor, 
                 n_samples: int = 20,
                 confidence_threshold: float = 0.5):
        """
        Args:
            lstm_model: Trained LSTM model
            n_samples: Number of MC dropout samples
            confidence_threshold: Threshold for making predictions
        """
        self.model = lstm_model
        self.n_samples = n_samples
        self.confidence_threshold = confidence_threshold
        
    def predict_with_confidence(self, df: pd.DataFrame, 
                                last_sequence_only: bool = True) -> Dict:
        """
        Make predictions with confidence scores using MC Dropout
        
        Returns:
            dict with:
                - predicted_prices: Mean predictions
                - confidence_scores: 1 / (1 + std_dev)
                - uncertainty: Raw standard deviation
                - should_predict: Boolean mask
                - predictions_selective: Only high-confidence predictions
        """
        if self.model.model is None:
            raise ValueError("Model not trained")
        
        # Prepare data
        from services.feature_service import get_feature_service, FeatureSet
        feature_service = get_feature_service()
        df_features, _ = feature_service.prepare_for_lstm(
            df, feature_set=FeatureSet.LSTM
        )
        
        df_clean = df_features[self.model.feature_columns + ['Close']].dropna()
        
        if len(df_clean) < self.model.sequence_length:
            raise ValueError(f"Need at least {self.model.sequence_length} data points")
        
        last_close = df_clean['Close'].iloc[-1]
        
        # Scale features
        features_scaled = self.model.feature_scaler.transform(
            df_clean[self.model.feature_columns].values
        )
        
        if last_sequence_only:
            X = features_scaled[-self.model.sequence_length:].reshape(
                1, self.model.sequence_length, -1
            )
        else:
            num_sequences = len(features_scaled) - self.model.sequence_length + 1
            X = np.zeros((num_sequences, self.model.sequence_length, features_scaled.shape[1]))
            for i in range(num_sequences):
                X[i] = features_scaled[i:i+self.model.sequence_length]
        
        X = torch.FloatTensor(X).to(self.model.device)
        
        # ====================================================================
        # MONTE CARLO DROPOUT: Sample multiple predictions with dropout on
        # ====================================================================
        self.model.model.train()  # Enable dropout at inference time
        
        predictions_scaled = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model.model(X)
                predictions_scaled.append(pred.cpu().numpy())
        
        predictions_scaled = np.array(predictions_scaled)  # (n_samples, batch_size, 1)
        
        # Calculate statistics
        mean_pred_scaled = predictions_scaled.mean(axis=0)
        std_pred_scaled = predictions_scaled.std(axis=0)
        
        # Inverse transform to get actual prices
        mean_pred = self.model.target_scaler.inverse_transform(mean_pred_scaled)
        
        # ====================================================================
        # CONFIDENCE CALCULATION
        # ====================================================================
        # Method 1: Inverse of standard deviation (lower std = higher confidence)
        # Normalize by mean to make it scale-independent
        relative_std = std_pred_scaled / (np.abs(mean_pred_scaled) + 1e-8)
        confidence = 1 / (1 + relative_std)
        
        # Should predict if confidence above threshold
        should_predict = confidence > self.confidence_threshold
        
        # Selective predictions (NaN where confidence low)
        predictions_selective = mean_pred.copy()
        predictions_selective[~should_predict] = np.nan
        
        return {
            'predicted_prices': mean_pred.flatten(),
            'confidence_scores': confidence.flatten(),
            'uncertainty': std_pred_scaled.flatten(),
            'should_predict': should_predict.flatten(),
            'predictions_selective': predictions_selective.flatten(),
            'last_close': last_close,
            'coverage': should_predict.mean(),
            'n_samples': self.n_samples
        }
    
    def predict(self, df: pd.DataFrame, last_sequence_only: bool = True):
        """
        Standard predict method (returns only high-confidence predictions)
        """
        result = self.predict_with_confidence(df, last_sequence_only)
        
        if last_sequence_only:
            if result['should_predict'][0]:
                return np.array([[result['predicted_prices'][0]]])
            else:
                return np.array([[np.nan]])
        else:
            return result['predictions_selective'].reshape(-1, 1)


def evaluate_with_confidence_threshold(model: LSTMStockPredictor, 
                                      df_test: pd.DataFrame,
                                      thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Evaluate model at different confidence thresholds to find optimal
    
    Args:
        model: Trained LSTM model
        df_test: Test data
        thresholds: List of confidence thresholds to try
        
    Returns:
        dict with results for each threshold
    """
    print(f"\n{'='*80}")
    print("CONFIDENCE THRESHOLD OPTIMIZATION")
    print(f"{'='*80}")
    
    results = []
    
    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold:.2f} ---")
        
        estimator = ConfidenceEstimator(
            model, 
            n_samples=20,
            confidence_threshold=threshold
        )
        
        # Get predictions
        result = estimator.predict_with_confidence(df_test, last_sequence_only=False)
        
        predictions = result['predicted_prices']
        confidence = result['confidence_scores']
        should_predict = result['should_predict']
        
        # Get actual prices
        actual_prices = df_test['Close'].values[-len(predictions):]
        prev_prices = df_test['Close'].shift(1).values[-len(predictions):]
        
        # Overall accuracy (treating abstentions as wrong)
        pred_dir = (predictions > prev_prices).astype(float)
        actual_dir = (actual_prices > prev_prices).astype(float)
        pred_dir[~should_predict] = 1 - actual_dir[~should_predict]
        overall_acc = (pred_dir == actual_dir).mean() * 100
        
        # Selective accuracy (only on predicted days)
        if should_predict.sum() > 0:
            pred_dir_sel = (predictions[should_predict] > prev_prices[should_predict]).astype(float)
            actual_dir_sel = (actual_prices[should_predict] > prev_prices[should_predict]).astype(float)
            selective_acc = (pred_dir_sel == actual_dir_sel).mean() * 100
        else:
            selective_acc = 0.0
        
        coverage = should_predict.mean() * 100
        
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Selective Accuracy: {selective_acc:.2f}%")
        print(f"  Overall Accuracy: {overall_acc:.2f}%")
        
        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'selective_acc': selective_acc,
            'overall_acc': overall_acc,
            'avg_confidence': confidence.mean(),
            'improvement': selective_acc - overall_acc
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Threshold':<12} {'Coverage':<12} {'Selective':<12} {'Overall':<12} {'Improvement'}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['coverage']:<12.1f} {r['selective_acc']:<12.2f} "
              f"{r['overall_acc']:<12.2f} {r['improvement']:+.2f}%")
    
    # Find best threshold
    best = max(results, key=lambda x: x['selective_acc'])
    print(f"\n🎯 Best threshold: {best['threshold']:.2f}")
    print(f"   Selective Accuracy: {best['selective_acc']:.2f}%")
    print(f"   Coverage: {best['coverage']:.1f}%")
    
    return results


if __name__ == "__main__":
    print("""
    Simple Confidence-Based Abstention
    ===================================
    
    This uses Monte Carlo Dropout to estimate uncertainty without
    changing the model architecture or loss function.
    
    Advantages:
    - Works with ANY trained LSTM
    - No retraining needed
    - Simple to implement and tune
    - Just add a threshold parameter
    
    Usage:
    ------
    from confidence_estimator import ConfidenceEstimator
    
    # Train standard LSTM
    model = LSTMStockPredictor(...)
    model.train(df_train)
    
    # Add confidence estimation
    estimator = ConfidenceEstimator(
        model,
        n_samples=20,
        confidence_threshold=0.5
    )
    
    # Predict with confidence
    result = estimator.predict_with_confidence(df_test)
    
    print(f"Coverage: {result['coverage']*100:.1f}%")
    print(f"Predictions: {result['predictions_selective']}")
    """)
"""
Test the Selective LSTM model against standard LSTM - FIXED VERSION

Fixes:
1. Proper data split with buffer (no leakage)
2. Proper feature engineering (features calculated once on full data)
3. Added noise to synthetic data (more realistic)
4. Fixed broken run_comparison_suite function

This script compares:
1. Standard LSTM (predicts on all days)
2. Selective LSTM (only predicts on confident days)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.selective_lstm_model import SelectiveLSTMPredictor
from models.lstm_model import LSTMStockPredictor
from test_lstm_synthetic_data import (
    generate_simple_trend, generate_mean_reversion, 
    generate_momentum, generate_complex_pattern, generate_random_walk
)


def add_noise(df, noise_level=0.02):
    """
    Add realistic noise to synthetic data
    
    Args:
        df: DataFrame with Close prices
        noise_level: Std dev of noise as fraction of price (default 2%)
    
    Returns:
        DataFrame with noisy prices
    """
    df = df.copy()
    prices = df['Close'].values
    
    # Add gaussian noise proportional to price
    noise = np.random.randn(len(prices)) * prices * noise_level
    df['Close'] = prices + noise
    
    # Ensure positive prices
    df['Close'] = np.maximum(df['Close'], 0.01)
    
    return df


def prepare_features(df):
    """
    PROPER feature engineering: Calculate features on entire dataset ONCE
    This prevents look-ahead bias
    
    Args:
        df: DataFrame with Close prices
        
    Returns:
        df_features: DataFrame with features
        feature_cols: List of feature column names
    """
    df = df.copy()
    
    # Technical indicators
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=10, min_periods=1).std()
    
    # Price ratios
    df['Price_SMA5_Ratio'] = df['Close'] / (df['SMA_5'] + 1e-8)
    df['Price_SMA20_Ratio'] = df['Close'] / (df['SMA_20'] + 1e-8)
    
    # Returns
    df['Return_1d'] = df['Close'].pct_change(1)
    
    feature_cols = [
        'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 
        'Momentum_5', 'Momentum_10', 'Volatility',
        'Price_SMA5_Ratio', 'Price_SMA20_Ratio', 'Return_1d'
    ]
    
    # Drop NaNs from indicators
    df = df.dropna()
    
    return df, feature_cols


def evaluate_selective_model(predictor, df_test, y_prev_test, pattern_name, actual_test_start=0):
    """
    Evaluate selective model with detailed metrics
    
    Args:
        predictor: Trained selective LSTM
        df_test: Test data (may include buffer)
        y_prev_test: Previous prices (only for actual test portion)
        pattern_name: Name of pattern
        actual_test_start: Index where actual test data starts (after buffer)
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {pattern_name}")
    print(f"{'='*80}")
    
    # Get predictions with confidence
    result = predictor.predict_with_confidence(df_test, last_sequence_only=False)
    
        # Get arrays (result may be a DataFrame or dict depending on predictor)
    predictions_all = result['predicted_prices']
    confidence_all = result['confidence_scores']
    should_predict_all = result['should_predict']

    # FIXED: Only evaluate on actual test portion (skip buffer)
    # Coerce to numpy arrays and ensure boolean mask for should_predict
    predictions = np.asarray(predictions_all[actual_test_start:]).astype(float)
    confidence = np.asarray(confidence_all[actual_test_start:]).astype(float)
    target_coverage = predictor.target_coverage
    
    # Clean up confidence scores (remove NaNs from buffer/reindexing)
    valid_mask = ~np.isnan(confidence)
    confidence_valid = confidence[valid_mask]
    
    if len(confidence_valid) == 0:
        # No valid predictions to evaluate
        print("Warning: No valid confidence scores found in test set.")
        selective_acc = 0.0
        selective_rmse = float('inf')
        coverage = 0.0
        should_predict = np.zeros_like(confidence, dtype=bool)
    else:
        # Find the confidence threshold that gives us the target coverage
        # e.g., for 0.5 coverage, we find the 50th percentile
        # We use (1.0 - target_coverage) because quantile(0.5) is the 50th percentile
        threshold = np.quantile(confidence_valid, 1.0 - target_coverage)
        
        # We predict on any day where confidence is >= this adaptive threshold
        # We must use the original 'confidence' array (with NaNs) to keep alignment
        should_predict = (confidence >= threshold)
    
    # Get actual prices (only test portion)
    actual_prices = df_test['Close'].values[actual_test_start:actual_test_start+len(predictions)]
    prev_prices = y_prev_test[:len(predictions)]
    
    # Overall metrics (treating abstentions as wrong)
    pred_direction = (predictions > prev_prices).astype(float)
    actual_direction = (actual_prices > prev_prices).astype(float)
    
    # Mark abstentions as wrong for overall accuracy
    pred_direction[~should_predict] = 1 - actual_direction[~should_predict]
    
    overall_acc = (pred_direction == actual_direction).mean() * 100
    
    # Selective metrics (only on predicted days)
    if should_predict.sum() > 0:
        predictions_selective = predictions[should_predict]
        actual_selective = actual_prices[should_predict]
        prev_selective = prev_prices[should_predict]
        
        pred_dir_selective = (predictions_selective > prev_selective).astype(float)
        actual_dir_selective = (actual_selective > prev_selective).astype(float)
        
        selective_acc = (pred_dir_selective == actual_dir_selective).mean() * 100
        
        # RMSE on predicted days
        selective_rmse = np.sqrt(((predictions_selective - actual_selective) ** 2).mean())
    else:
        selective_acc = 0.0
        selective_rmse = float('inf')
    
    # Coverage
    coverage = should_predict.mean() * 100
    
    # Abstention quality
    if should_predict.sum() > 0 and (~should_predict).sum() > 0:
        error_predicted = np.abs(predictions[should_predict] - actual_prices[should_predict]).mean()
        error_abstained = np.abs(predictions[~should_predict] - actual_prices[~should_predict]).mean()
        abstention_quality = error_abstained / error_predicted
    else:
        abstention_quality = np.nan
    
    print(f"\n📊 Results:")
    print(f"  Coverage: {coverage:.1f}% ({should_predict.sum()}/{len(should_predict)} days)")
    print(f"  Avg Confidence: {confidence.mean():.3f}")
    print(f"  ")
    print(f"  Overall Accuracy: {overall_acc:.2f}% (all days, abstentions = wrong)")
    print(f"  Selective Accuracy: {selective_acc:.2f}% (only predicted days)")
    print(f"  Selective RMSE: ${selective_rmse:.2f}")
    print(f"  ")
    print(f"  Abstention Quality: {abstention_quality:.2f}x")
    print(f"    (Error on abstained days / Error on predicted days)")
    print(f"    (>1.0 means model correctly identified harder days)")
    
    return {
        'pattern': pattern_name,
        'overall_acc': overall_acc,
        'selective_acc': selective_acc,
        'coverage': coverage,
        'avg_confidence': confidence.mean(),
        'selective_rmse': selective_rmse,
        'abstention_quality': abstention_quality,
        'passed_selective': selective_acc > 60.0
    }


def compare_models(pattern_name, df_raw, config_standard, config_selective):
    """
    Compare standard LSTM vs selective LSTM on same data
    
    FIXED: Proper feature engineering and data split
    """
    print(f"\n\n{'#'*80}")
    print(f"# COMPARING MODELS: {pattern_name}")
    print(f"{'#'*80}")
    
    # Get sequence length from config
    sequence_length = config_standard['sequence_length']
    
    # STEP 1: Calculate features on ENTIRE dataset (no leakage - features use only past data)
    print("\n📊 Preparing features...")
    df, feature_cols = prepare_features(df_raw)
    print(f"  Original data: {len(df_raw)} rows")
    print(f"  After features: {len(df)} rows (dropped {len(df_raw)-len(df)} NaN rows)")
    print(f"  Features: {len(feature_cols)}")
    
    # STEP 2: Split data AFTER feature engineering
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]
    
    # FIXED: Test set needs buffer from val/train for sequence input
    test_start_idx = train_size + val_size
    df_test_with_buffer = df.iloc[test_start_idx - sequence_length:]
    
    # Mark where actual test data starts (after buffer)
    actual_test_start = sequence_length
    
    print(f"\n📊 Data split:")
    print(f"  Train: {len(df_train)} rows (indices 0-{train_size-1})")
    print(f"  Val: {len(df_val)} rows (indices {train_size}-{train_size+val_size-1})")
    print(f"  Test: {len(df_test_with_buffer)-sequence_length} rows + {sequence_length} buffer")
    
    # Get previous prices for direction calculation (only for actual test portion)
    y_prev_test = df_test_with_buffer['Close'].shift(1).values[actual_test_start:]
    
    # ============================================================================
    # 1. STANDARD LSTM
    # ============================================================================
    print(f"\n{'='*80}")
    print("TRAINING STANDARD LSTM")
    print(f"{'='*80}")
    
    standard_lstm = LSTMStockPredictor(**config_standard)
    standard_lstm.train(
        df_train,
        epochs=50,
        batch_size=32,
        validation_sequences=50,
        use_validation=True,
        early_stopping_patience=15
    )
    
    # FIXED: Predict on data with buffer, but only evaluate on actual test portion
    predictions_all = standard_lstm.predict(df_test_with_buffer, last_sequence_only=False).flatten()
    
    # Only keep predictions for actual test data (skip buffer)
    predictions_standard = predictions_all[actual_test_start:]
    
    # Get corresponding actual prices (only test portion)
    actual_prices = df_test_with_buffer['Close'].values[actual_test_start:actual_test_start+len(predictions_standard)]
    prev_prices = y_prev_test[:len(predictions_standard)]
    
    # Debug: Check for leakage
    print(f"\n🔍 Data Leakage Check:")
    print(f"  Training data ends at index: {train_size-1}")
    print(f"  Test predictions start at actual index: {test_start_idx}")
    print(f"  Buffer used (for sequences): {sequence_length} rows")
    print(f"  Actual test predictions: {len(predictions_standard)} days")
    print(f"  First prediction uses indices: {test_start_idx - sequence_length} to {test_start_idx-1}")
    print(f"  (Buffer overlaps with validation, NOT training) ✓")
    
    pred_dir = (predictions_standard > prev_prices).astype(float)
    actual_dir = (actual_prices > prev_prices).astype(float)
    standard_acc = (pred_dir == actual_dir).mean() * 100
    
    print(f"\n📊 Standard LSTM Results:")
    print(f"  Direction Accuracy: {standard_acc:.2f}%")
    print(f"  Coverage: 100% (predicts all days)")
    
    # ============================================================================
    # 2. SELECTIVE LSTM
    # ============================================================================
    print(f"\n{'='*80}")
    print("TRAINING SELECTIVE LSTM")
    print(f"{'='*80}")
    
    selective_lstm = SelectiveLSTMPredictor(**config_selective)
    selective_lstm.train(
        df_train,
        epochs=50,
        batch_size=32,
        validation_sequences=50,
        use_validation=True,
        early_stopping_patience=20
    )
    
    # FIXED: Use same test data with buffer
    selective_results = evaluate_selective_model(
        selective_lstm, df_test_with_buffer, y_prev_test, pattern_name, 
        actual_test_start=actual_test_start
    )
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    print(f"\n{'='*80}")
    print("📊 COMPARISON")
    print(f"{'='*80}")
    print(f"\nStandard LSTM:")
    print(f"  Direction Accuracy: {standard_acc:.2f}%")
    print(f"  Coverage: 100%")
    print(f"")
    print(f"Selective LSTM:")
    print(f"  Overall Accuracy: {selective_results['overall_acc']:.2f}%")
    print(f"  Selective Accuracy: {selective_results['selective_acc']:.2f}%")
    print(f"  Coverage: {selective_results['coverage']:.1f}%")
    print(f"")
    
    # Improvement
    improvement = selective_results['selective_acc'] - standard_acc
    print(f"💡 Improvement: {improvement:+.2f}% on predicted days")
    
    if improvement > 5:
        print(f"   ✅ Significant improvement by selective prediction!")
    elif improvement > 0:
        print(f"   ✓ Modest improvement")
    else:
        print(f"   ⚠ No improvement - selective strategy not helping")
    
    return {
        'pattern': pattern_name,
        'standard_acc': standard_acc,
        'selective_overall': selective_results['overall_acc'],
        'selective_acc': selective_results['selective_acc'],
        'coverage': selective_results['coverage'],
        'improvement': improvement
    }


def run_comparison_suite():
    """
    Run full comparison across multiple patterns
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       SELECTIVE LSTM vs STANDARD LSTM COMPARISON                ║
╚══════════════════════════════════════════════════════════════════╝

This will test whether selective prediction improves accuracy
by abstaining on difficult-to-predict days.

FIXES APPLIED:
✓ Proper data split (no leakage)
✓ Proper feature engineering
✓ Added 2% noise to synthetic data
✓ Buffer handling for test sequences
""")
    
    # Configurations
    config_standard = {
        'sequence_length': 20,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'bidirectional': True,
        'use_layer_norm': False,
        'use_residual': False
    }
    
    config_selective = {
        'sequence_length': 20,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'bidirectional': True,
        'use_layer_norm': False,
        'use_residual': False,
        'target_coverage': 0.10,
        'confidence_threshold': 0.5
    }
    
    # Test patterns with added noise for realism
    print("\n🎲 Generating synthetic data with 2% noise...")
    patterns = [
        ('Momentum', lambda: add_noise(generate_momentum(n_days=400), noise_level=0.02)),
        ('Mean Reversion', lambda: add_noise(generate_mean_reversion(n_days=400), noise_level=0.02)),
        ('Complex Pattern', lambda: add_noise(generate_complex_pattern(n_days=400), noise_level=0.02)),
    ]
    
    results = []
    
    for pattern_name, generator in patterns:
        # Generate fresh data
        df = generator()
        
        # CRITICAL FIX: Force garbage collection between patterns
        # This ensures models don't carry over state
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = compare_models(pattern_name, df, config_standard, config_selective)
        results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for r in results:
        print(f"{r['pattern']:20s}:")
        print(f"  Standard: {r['standard_acc']:5.1f}% | Selective: {r['selective_acc']:5.1f}% | Coverage: {r['coverage']:5.1f}%")
        print(f"  Improvement: {r['improvement']:+.1f}%")
        print()
    
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_coverage = np.mean([r['coverage'] for r in results])
    
    print(f"Average Improvement: {avg_improvement:+.1f}%")
    print(f"Average Coverage: {avg_coverage:.1f}%")
    print()
    
    if avg_improvement > 5:
        print("✅ SUCCESS! Selective prediction significantly improves accuracy!")
        print("   The model learned to abstain on difficult days.")
    elif avg_improvement > 0:
        print("✓ Selective prediction helps modestly.")
        print("   Consider tuning target_coverage or confidence_threshold.")
    else:
        print("⚠ Selective prediction not helping.")
        print("   Possible issues:")
        print("   - Model not learning good confidence calibration")
        print("   - Need more training data")
        print("   - Need to adjust loss function weights")
        print("\n💡 Expected accuracies with proper fixes:")
        print("   - Standard LSTM: 55-75% (not 95%+)")
        print("   - Selective LSTM: Higher than standard on predicted days")
        print("   - Coverage: 65-75% (not 100%)")


if __name__ == "__main__":
    run_comparison_suite()
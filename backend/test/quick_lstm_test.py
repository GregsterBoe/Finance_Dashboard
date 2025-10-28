"""
Quick Start: Test Your LSTM in 5 Minutes
=========================================

This script will immediately test if your LSTM is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_lstm_synthetic_data import *
from models.lstm_model import LSTMStockPredictor

print("""
╔══════════════════════════════════════════════════════════════════╗
║            QUICK LSTM TEST - 5 MINUTE DIAGNOSTIC                ║
╚══════════════════════════════════════════════════════════════════╝

This will test your LSTM on synthetic data with KNOWN patterns.
If it can't learn these simple patterns, something is wrong!
""")

# ============================================================================
# TEST 1: Simple Momentum Pattern (Easiest)
# ============================================================================

print("\n" + "="*70)
print("TEST 1: MOMENTUM PATTERN (Easiest - Yesterday predicts today)")
print("="*70)

print("\nGenerating synthetic data...")
df = generate_momentum(n_days=300, start_price=100, momentum_strength=0.6)

print(f"✓ Generated {len(df)} days")
print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

# Calculate what the pattern looks like
returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
returns_yesterday = returns.shift(1).dropna()
returns_today = returns[returns_yesterday.index]

# Correlation between yesterday and today
correlation = np.corrcoef(returns_yesterday, returns_today)[0, 1]
print(f"  Pattern strength: {correlation:.3f} correlation between consecutive returns")
print(f"  (If LSTM learns this, direction accuracy should be >60%)")

# Split data
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print(f"\n✓ Split: {len(df_train)} train, {len(df_test)} test")

# Create and train LSTM
print("\n" + "-"*70)
print("Training LSTM...")
print("-"*70)

predictor = LSTMStockPredictor(
    sequence_length=10,  # Short sequence for quick test
    hidden_size=32,       # Small model for quick test
    num_layers=1,         # Single layer for quick test
    dropout=0.1,
    learning_rate=0.001
)

try:
    history = predictor.train(
        df_train,
        epochs=30,  # Quick training
        batch_size=32,
        validation_sequences=20,
        use_validation=True,
        early_stopping_patience=10
    )
    
    print("\n✓ Training completed!")
    
    # Check if loss decreased
    if len(history['train_loss']) > 0:
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"\nLoss Improvement:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        
        if improvement > 10:
            print("  ✓ Model is learning (loss decreased >10%)")
        else:
            print("  ⚠ WARNING: Model may not be learning well (loss decreased <10%)")
    
    # Get validation metrics
    print("\n" + "-"*70)
    print("Validation Metrics:")
    print("-"*70)
    
    val_metrics = predictor.get_validation_metrics()
    
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  MAE: {val_metrics['mae']:.6f}")
    print(f"  R²: {val_metrics['r2_score']:.4f}")
    print(f"  Direction Accuracy: {val_metrics['direction_accuracy']:.2f}%")
    
    # Test on test set
    print("\n" + "-"*70)
    print("Testing on held-out test set...")
    print("-"*70)
    
    predictions = []
    actuals = []
    
    # Make predictions
    for i in range(min(30, len(df_test) - 1)):  # Test on 30 days
        df_pred = pd.concat([df_train, df_test.iloc[:i+1]])
        
        try:
            result = predictor.predict_with_price_conversion(df_pred, last_sequence_only=True)
            predictions.append(result['predicted_return'])
            
            actual_return = np.log(df_test.iloc[i+1]['Close'] / df_test.iloc[i]['Close'])
            actuals.append(actual_return)
        except Exception as e:
            print(f"  Prediction {i} failed: {e}")
            break
    
    if len(predictions) > 0:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        
        test_mae = mean_absolute_error(actuals, predictions)
        test_r2 = r2_score(actuals, predictions)
        
        # Direction accuracy
        correct_direction = np.sum((actuals > 0) == (predictions > 0))
        direction_accuracy = (correct_direction / len(actuals)) * 100
        
        print(f"\nTest Set Results ({len(predictions)} predictions):")
        print(f"  MAE: {test_mae:.6f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  Direction Accuracy: {direction_accuracy:.2f}%")
        
        # Interpretation
        print("\n" + "="*70)
        print("RESULT INTERPRETATION:")
        print("="*70)
        
        if direction_accuracy >= 60:
            print("✓✓✓ EXCELLENT! Your LSTM is working correctly!")
            print(f"    Direction accuracy {direction_accuracy:.1f}% > 60% on momentum pattern")
            print("    The model learned that yesterday predicts today.")
        elif direction_accuracy >= 55:
            print("✓ GOOD! Your LSTM is learning.")
            print(f"    Direction accuracy {direction_accuracy:.1f}% > 55%")
            print("    Better than random (50%), but could improve with:")
            print("    - More training epochs")
            print("    - Larger model (hidden_size, num_layers)")
            print("    - More training data")
        elif direction_accuracy >= 50:
            print("⚠ MARGINAL. Your LSTM is barely learning.")
            print(f"    Direction accuracy {direction_accuracy:.1f}% ≈ 50% (random)")
            print("    Possible issues:")
            print("    - Model not training long enough")
            print("    - Learning rate too high/low")
            print("    - Model too small")
        else:
            print("✗ PROBLEM! Your LSTM is NOT working correctly!")
            print(f"    Direction accuracy {direction_accuracy:.1f}% < 50% (worse than random!)")
            print("    Likely issues:")
            print("    - Data leakage in features")
            print("    - Scaling issues (fit_transform vs transform)")
            print("    - Feature/target misalignment")
            print("    - Model not actually training")
        
        # Show some predictions
        print("\n" + "-"*70)
        print("Sample Predictions (first 10):")
        print("-"*70)
        print(f"{'Actual':>10} {'Predicted':>10} {'Direction':>12} {'Correct':>8}")
        print("-"*70)
        
        for i in range(min(10, len(predictions))):
            actual = actuals[i]
            pred = predictions[i]
            actual_dir = "UP ↑" if actual > 0 else "DOWN ↓"
            pred_dir = "UP ↑" if pred > 0 else "DOWN ↓"
            correct = "✓" if (actual > 0) == (pred > 0) else "✗"
            
            print(f"{actual:10.6f} {pred:10.6f} {actual_dir:>6} → {pred_dir:<6} {correct:>4}")
        
        # Detailed diagnostics if failed
        if direction_accuracy < 55:
            print("\n" + "="*70)
            print("DIAGNOSTIC CHECKS:")
            print("="*70)
            
            # Check 1: Are predictions all the same?
            pred_std = np.std(predictions)
            if pred_std < 0.001:
                print("✗ All predictions are nearly identical!")
                print("  Issue: Model is not learning, outputting constant value")
                print("  Possible causes:")
                print("    - Model stuck in local minimum")
                print("    - Learning rate too low")
                print("    - Vanishing gradients")
            else:
                print(f"✓ Predictions have variance (std={pred_std:.6f})")
            
            # Check 2: Are predictions centered around zero?
            pred_mean = np.mean(predictions)
            if abs(pred_mean) > 0.01:
                print(f"⚠ Predictions biased: mean={pred_mean:.6f}")
                print("  Model is predicting consistently up or down")
            else:
                print(f"✓ Predictions centered (mean={pred_mean:.6f})")
            
            # Check 3: Feature information
            print(f"\n✓ Feature columns used: {predictor.feature_columns}")
            
            print("\n" + "-"*70)
            print("RECOMMENDED ACTIONS:")
            print("-"*70)
            print("1. Run the full test suite:")
            print("   from test_lstm_synthetic_data import run_full_test_suite")
            print("   results = run_full_test_suite(LSTMStockPredictor, ...)")
            print("")
            print("2. Check your feature_service.py:")
            print("   - Is target correctly calculated?")
            print("   - Any data leakage in features?")
            print("")
            print("3. Check your scaling:")
            print("   - Using transform() (not fit_transform()) on test data?")
            print("   - Saving and loading scaler correctly?")
    
    else:
        print("✗ ERROR: Could not make predictions on test set!")
        print("  Check error messages above.")

except Exception as e:
    print(f"\n✗ ERROR during training/testing:")
    print(f"  {type(e).__name__}: {str(e)}")
    print("\nStack trace:")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*70)
    print("DEBUGGING SUGGESTIONS:")
    print("="*70)
    print("1. Check your import path for LSTMStockPredictor")
    print("2. Ensure all dependencies are installed")
    print("3. Check if feature_service.py is accessible")
    print("4. Run the diagnostic script: python scaling_comprehensive_guide.py")

print("\n" + "="*70)
print("QUICK TEST COMPLETE")
print("="*70)
print("\nNext steps:")
print("- If test passed: Your LSTM is working! Try on real data.")
print("- If test failed: Run full test suite for detailed diagnostics")
print("- See test_lstm_synthetic_data.py for full testing framework")
"""
LSTM Testing Suite with Synthetic Data
=======================================

This creates artificial stock data with KNOWN patterns so you can verify
your LSTM is learning correctly. We'll create several patterns:

1. Simple Trend - Stock goes up consistently
2. Mean Reversion - Stock oscillates around a mean
3. Momentum - Yesterday's return predicts today's return
4. Complex Pattern - Combination of patterns

If your LSTM can't learn these simple patterns, something is wrong!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================================================================
# Pattern Generators
# ============================================================================

def generate_simple_trend(n_days=500, start_price=100, trend=0.001):
    """
    Generate stock data with simple upward trend.
    
    Pattern: Each day goes up by trend% + small noise
    LSTM should easily predict this!
    
    Args:
        n_days: Number of days to generate
        start_price: Starting stock price
        trend: Daily trend (0.001 = 0.1% daily growth)
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate returns with trend + noise
    returns = np.random.normal(trend, 0.005, n_days)  # Small noise
    
    # Calculate prices
    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_mean_reversion(n_days=500, start_price=100, mean_price=100, reversion_speed=0.1):
    """
    Generate stock data with mean reversion.
    
    Pattern: Stock always moves back toward mean price
    If price > mean, next return is negative
    If price < mean, next return is positive
    
    LSTM should learn: "When far from mean, predict return toward mean"
    
    Args:
        n_days: Number of days
        start_price: Starting price
        mean_price: Price to revert to
        reversion_speed: How fast it reverts (0.1 = moderate)
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    prices = [start_price]
    
    for i in range(1, n_days):
        # Calculate distance from mean
        distance = prices[-1] - mean_price
        
        # Return is proportional to distance (negative to revert)
        mean_return = -reversion_speed * (distance / mean_price)
        
        # Add noise
        return_val = mean_return + np.random.normal(0, 0.01)
        
        # Calculate next price
        next_price = prices[-1] * (1 + return_val)
        prices.append(next_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_momentum(n_days=500, start_price=100, momentum_strength=0.5):
    """
    Generate stock data with momentum.
    
    Pattern: Yesterday's return predicts today's return
    If yesterday was +2%, today is likely +1% (half the momentum)
    
    LSTM should learn: "Use yesterday's return to predict today's"
    
    Args:
        n_days: Number of days
        start_price: Starting price
        momentum_strength: How much momentum carries over (0.5 = half)
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    returns = [np.random.normal(0, 0.02)]  # First return is random
    
    for i in range(1, n_days):
        # Today's return = momentum_strength * yesterday's return + noise
        momentum_return = momentum_strength * returns[-1]
        noise = np.random.normal(0, 0.01)
        returns.append(momentum_return + noise)
    
    # Calculate prices from returns
    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_complex_pattern(n_days=500, start_price=100):
    """
    Generate stock data with complex but learnable pattern.
    
    Pattern: Combination of:
    - Weekly cycle (Monday up, Friday down)
    - Momentum (yesterday affects today)
    - Trend (slight upward bias)
    
    This tests if LSTM can learn multiple patterns simultaneously.
    
    Args:
        n_days: Number of days
        start_price: Starting price
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    prices = [start_price]
    returns = [0]
    
    for i in range(1, n_days):
        # Component 1: Weekly cycle
        day_of_week = dates[i].dayofweek
        if day_of_week == 0:  # Monday
            weekly_effect = 0.005
        elif day_of_week == 4:  # Friday
            weekly_effect = -0.003
        else:
            weekly_effect = 0
        
        # Component 2: Momentum
        momentum_effect = 0.3 * returns[-1]
        
        # Component 3: Trend
        trend_effect = 0.0005
        
        # Component 4: Noise
        noise = np.random.normal(0, 0.01)
        
        # Total return
        return_val = weekly_effect + momentum_effect + trend_effect + noise
        returns.append(return_val)
        
        # Calculate price
        next_price = prices[-1] * (1 + return_val)
        prices.append(next_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_random_walk(n_days=500, start_price=100, volatility=0.02):
    """
    Generate random walk data (NO PATTERN).
    
    Pattern: NONE! Pure random walk.
    Each day's return is completely random.
    
    LSTM should NOT be able to predict this well!
    If it does, you likely have data leakage.
    
    Args:
        n_days: Number of days
        start_price: Starting price
        volatility: Daily volatility (0.02 = 2%)
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Pure random returns - NO PATTERN
    returns = np.random.normal(0, volatility, n_days)
    
    # Calculate prices
    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_sideways_market(n_days=500, start_price=100, range_pct=0.05, cycles=5):
    """
    Generate sideways/range-bound market data.
    
    Pattern: Stock oscillates within a range, no clear trend.
    Price bounces between support and resistance levels.
    
    This is challenging for trend-following models!
    
    Args:
        n_days: Number of days
        start_price: Starting price
        range_pct: Range as percentage of price (0.05 = ±5%)
        cycles: Number of complete cycles through the range
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Create oscillating pattern within range
    support = start_price * (1 - range_pct)
    resistance = start_price * (1 + range_pct)
    
    prices = [start_price]
    
    # Create sine wave pattern for oscillation
    frequency = 2 * np.pi * cycles / n_days
    
    for i in range(1, n_days):
        # Sine wave oscillation
        sine_value = np.sin(frequency * i)
        
        # Map sine [-1, 1] to [support, resistance]
        target_price = support + (sine_value + 1) / 2 * (resistance - support)
        
        # Add some momentum and noise
        current_price = prices[-1]
        direction = target_price - current_price
        
        # Move toward target with some noise
        return_val = direction / current_price * 0.3 + np.random.normal(0, 0.01)
        
        next_price = current_price * (1 + return_val)
        prices.append(next_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_volatile_market(n_days=500, start_price=100, volatility=0.03, mean_reversion=0.05):
    """
    Generate high volatility market data.
    
    Pattern: Large price swings with some mean reversion.
    Tests if LSTM can handle extreme movements.
    
    Args:
        n_days: Number of days
        start_price: Starting price
        volatility: Daily volatility (0.03 = 3% - very high)
        mean_reversion: Strength of mean reversion (0.05 = weak)
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    prices = [start_price]
    mean_price = start_price
    
    for i in range(1, n_days):
        # High volatility random component
        random_return = np.random.normal(0, volatility)
        
        # Weak mean reversion component
        distance = (prices[-1] - mean_price) / mean_price
        reversion_return = -mean_reversion * distance
        
        # Combine
        total_return = random_return + reversion_return
        
        next_price = prices[-1] * (1 + total_return)
        prices.append(next_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data with extra wide high-low range
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'High': prices * (1 + np.random.uniform(0.01, 0.03, n_days)),  # Wider range
        'Low': prices * (1 + np.random.uniform(-0.03, -0.01, n_days)),  # Wider range
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


def generate_random_walk(n_days=500, start_price=100):
    """
    Generate purely random data (NO pattern).
    
    Pattern: NONE - completely random
    LSTM should NOT be able to predict this well.
    
    If your LSTM gets high accuracy on this, something is wrong!
    (Likely data leakage)
    
    Args:
        n_days: Number of days
        start_price: Starting price
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Purely random returns
    returns = np.random.normal(0, 0.02, n_days)
    
    # Calculate prices
    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.002, 0.002, n_days)),
        'High': prices * (1 + np.random.uniform(0.002, 0.01, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.01, -0.002, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Dividends': np.zeros(n_days),
        'Stock Splits': np.zeros(n_days)
    })
    
    df.set_index('Date', inplace=True)
    
    return df


# ============================================================================
# Testing Functions
# ============================================================================

def test_lstm_on_pattern(lstm_predictor, pattern_name, df, expected_behavior):
    """
    Test LSTM on a specific pattern and check if it behaves correctly.
    
    Args:
        lstm_predictor: Your LSTMStockPredictor instance
        pattern_name: Name of the pattern being tested
        df: DataFrame with synthetic data
        expected_behavior: Dict with expected results
    
    Returns:
        Dict with test results
    """
    print("\n" + "=" * 80)
    print(f"Testing LSTM on: {pattern_name}")
    print("=" * 80)
    
    # Split data
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print(f"\nTrain size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Train model
    print("\nTraining model...")
    history = lstm_predictor.train(
        df_train,
        epochs=50,  # Reduced for faster testing
        batch_size=32,
        validation_sequences=30,
        use_validation=True,
        early_stopping_patience=10
    )
    
    # Get validation metrics
    val_metrics = lstm_predictor.get_validation_metrics()
    
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  MAE: {val_metrics['mae']:.6f}")
    print(f"  R²: {val_metrics['r2_score']:.4f}")
    print(f"  Direction Accuracy: {val_metrics['direction_accuracy']:.2f}%")
    
    # Make predictions on test set
    predictions = []
    actuals = []
    
    for i in range(len(df_test) - 1):
        # Get data up to this point
        df_pred = pd.concat([df_train, df_test.iloc[:i+1]])
        
        # Predict
        result = lstm_predictor.predict_with_price_conversion(df_pred, last_sequence_only=True)
        predictions.append(result['predicted_return'])
        
        # Get actual
        actual_return = np.log(df_test.iloc[i+1]['Close'] / df_test.iloc[i]['Close'])
        actuals.append(actual_return)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate test metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    test_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    test_mae = mean_absolute_error(actuals, predictions)
    test_r2 = r2_score(actuals, predictions)
    
    # Direction accuracy
    correct_direction = np.sum((actuals > 0) == (predictions > 0))
    direction_accuracy = (correct_direction / len(actuals)) * 100
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²: {test_r2:.4f}")
    print(f"  Direction Accuracy: {direction_accuracy:.2f}%")
    
    # Check against expected behavior
    print(f"\nExpected Behavior:")
    print(f"  {expected_behavior['description']}")
    
    results = {
        'pattern': pattern_name,
        'val_rmse': val_metrics['rmse'],
        'val_direction_acc': val_metrics['direction_accuracy'],
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_direction_acc': direction_accuracy,
        'expected': expected_behavior,
        'predictions': predictions,
        'actuals': actuals
    }
    
    # Evaluate if model met expectations
    if 'min_direction_acc' in expected_behavior:
        if direction_accuracy >= expected_behavior['min_direction_acc']:
            print(f"  ✓ PASSED: Direction accuracy {direction_accuracy:.1f}% >= {expected_behavior['min_direction_acc']}%")
            results['passed'] = True
        else:
            print(f"  ✗ FAILED: Direction accuracy {direction_accuracy:.1f}% < {expected_behavior['min_direction_acc']}%")
            results['passed'] = False
    
    if 'max_direction_acc' in expected_behavior:
        if direction_accuracy <= expected_behavior['max_direction_acc']:
            print(f"  ✓ PASSED: Direction accuracy {direction_accuracy:.1f}% <= {expected_behavior['max_direction_acc']}%")
            results['passed'] = results.get('passed', True)
        else:
            print(f"  ✗ SUSPICIOUS: Direction accuracy {direction_accuracy:.1f}% > {expected_behavior['max_direction_acc']}% (possible data leakage!)")
            results['passed'] = False
    
    return results


def visualize_predictions(results, save_path=None):
    """
    Visualize prediction results.
    
    Args:
        results: Results dict from test_lstm_on_pattern
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Actual vs Predicted Returns
    axes[0].plot(results['actuals'], label='Actual Returns', alpha=0.7)
    axes[0].plot(results['predictions'], label='Predicted Returns', alpha=0.7)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_title(f"{results['pattern']} - Actual vs Predicted Returns")
    axes[0].set_xlabel('Test Sample')
    axes[0].set_ylabel('Log Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction Error
    errors = results['predictions'] - results['actuals']
    axes[1].plot(errors, color='red', alpha=0.6, label='Prediction Error')
    axes[1].axhline(y=0, color='k', linestyle='--')
    axes[1].fill_between(range(len(errors)), errors, 0, alpha=0.3, color='red')
    axes[1].set_title('Prediction Errors')
    axes[1].set_xlabel('Test Sample')
    axes[1].set_ylabel('Error (Predicted - Actual)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Testing Suite
# ============================================================================

def run_full_test_suite(lstm_predictor_class, **lstm_kwargs):
    """
    Run complete LSTM test suite on all patterns.
    
    Args:
        lstm_predictor_class: Your LSTMStockPredictor class
        **lstm_kwargs: Arguments to pass to LSTMStockPredictor constructor
    
    Returns:
        Dict with all test results
    """
    print("=" * 80)
    print("LSTM TESTING SUITE - SYNTHETIC DATA")
    print("=" * 80)
    
    test_cases = [
        {
            'name': 'Simple Trend',
            'generator': lambda: generate_simple_trend(n_days=400),
            'expected': {
                'description': 'Stock has consistent upward trend. LSTM should predict positive returns.',
                'min_direction_acc': 55.0  # Should beat random (50%)
            }
        },
        {
            'name': 'Mean Reversion',
            'generator': lambda: generate_mean_reversion(n_days=400),
            'expected': {
                'description': 'Stock reverts to mean. LSTM should learn: high price → negative return.',
                'min_direction_acc': 60.0  # Clear pattern, should do well
            }
        },
        {
            'name': 'Momentum',
            'generator': lambda: generate_momentum(n_days=400, momentum_strength=0.5),
            'expected': {
                'description': 'Yesterday predicts today (momentum). LSTM should learn this easily.',
                'min_direction_acc': 65.0  # Very clear pattern
            }
        },
        {
            'name': 'Complex Pattern',
            'generator': lambda: generate_complex_pattern(n_days=400),
            'expected': {
                'description': 'Multiple patterns combined. LSTM should learn with more effort.',
                'min_direction_acc': 55.0
            }
        },
        {
            'name': 'Random Walk',
            'generator': lambda: generate_random_walk(n_days=400),
            'expected': {
                'description': 'NO PATTERN. LSTM should NOT predict well. ~50% direction accuracy expected.',
                'max_direction_acc': 60.0  # If higher, possible data leakage!
            }
        }
    ]
    
    all_results = []
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Generating data: {test_case['name']}")
        print(f"{'='*80}")
        
        # Generate data
        df = test_case['generator']()
        
        print(f"Data generated: {len(df)} days")
        print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        # Create new predictor for each test
        predictor = lstm_predictor_class(**lstm_kwargs)
        
        # Run test
        results = test_lstm_on_pattern(
            predictor,
            test_case['name'],
            df,
            test_case['expected']
        )
        
        all_results.append(results)
        
        # Visualize
        visualize_predictions(results, save_path=f"test_{test_case['name'].replace(' ', '_').lower()}.png")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    for result in all_results:
        status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
        print(f"\n{status} - {result['pattern']}")
        print(f"  Direction Accuracy: {result['test_direction_acc']:.2f}%")
        print(f"  R²: {result['test_r2']:.4f}")
    
    # Overall assessment
    passed = sum(1 for r in all_results if r.get('passed', False))
    total = len(all_results)
    
    print(f"\n{'='*80}")
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED! Your LSTM is working correctly!")
    elif passed >= total * 0.6:
        print("⚠ PARTIAL SUCCESS. Some patterns not learned well. Check model architecture or training.")
    else:
        print("✗ MULTIPLE FAILURES. Your LSTM may have implementation issues:")
        print("  - Check feature engineering (data leakage?)")
        print("  - Check scaling (using transform vs fit_transform correctly?)")
        print("  - Check model architecture")
        print("  - Check if model is actually training (loss decreasing?)")
    
    return all_results


# ============================================================================
# Quick Start Example
# ============================================================================

if __name__ == "__main__":
    print("""
    LSTM Testing Suite - Quick Start
    =================================
    
    This will generate synthetic data with known patterns and test your LSTM.
    
    To use:
    
    from models.lstm_model import LSTMStockPredictor
    from test_lstm_synthetic_data import run_full_test_suite
    
    # Run tests
    results = run_full_test_suite(
        LSTMStockPredictor,
        sequence_length=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )
    
    The test will:
    1. Generate 5 different patterns
    2. Train your LSTM on each
    3. Check if it learns the patterns
    4. Generate plots showing predictions
    5. Give you a pass/fail for each pattern
    
    Expected Results:
    -----------------
    ✓ Trend Pattern: Should pass (>55% direction accuracy)
    ✓ Mean Reversion: Should pass (>60% direction accuracy)
    ✓ Momentum: Should pass (>65% direction accuracy)
    ✓ Complex: Should pass (>55% direction accuracy)
    ✓ Random Walk: Should "fail" (~50% direction accuracy - this is correct!)
    
    If your LSTM passes all tests, it's working correctly!
    """)
    
    # Example: Generate one pattern to visualize
    print("\nGenerating example data...")
    df = generate_momentum(n_days=200)
    
    print(f"\nGenerated {len(df)} days of momentum pattern data")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print("\nSample data:")
    print(df.head(10))
    
    # Save example
    df.to_csv('synthetic_momentum_example.csv')
    print("\nSaved to: synthetic_momentum_example.csv")
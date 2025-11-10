"""
Test the Simple Confidence Estimator (MC Dropout approach)

This tests whether simply using dropout-based uncertainty
can provide selective prediction without model changes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.confidence_estimator import ConfidenceEstimator, evaluate_with_confidence_threshold
from models.lstm_model import LSTMStockPredictor
from test_lstm_synthetic_data import generate_momentum, generate_mean_reversion, generate_complex_pattern


def test_confidence_estimator(pattern_name, df):
    """
    Test confidence estimator on a pattern
    """
    print(f"\n\n{'#'*80}")
    print(f"# TESTING: {pattern_name}")
    print(f"{'#'*80}")
    
    # Split data
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print(f"Data: {len(df_train)} train, {len(df_test)} test")
    
    # Train standard LSTM
    print(f"\n{'='*80}")
    print("TRAINING STANDARD LSTM")
    print(f"{'='*80}")
    
    model = LSTMStockPredictor(
        sequence_length=20,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        bidirectional=True,
        use_layer_norm=False,
        use_residual=False
    )
    
    model.train(
        df_train,
        epochs=50,
        batch_size=32,
        validation_sequences=50,
        use_validation=True,
        early_stopping_patience=15
    )
    
    # Baseline: Standard predictions (no confidence)
    print(f"\n{'='*80}")
    print("BASELINE: Standard Predictions")
    print(f"{'='*80}")
    
    predictions_std = model.predict(df_test, last_sequence_only=False).flatten()
    actual_prices = df_test['Close'].values[-len(predictions_std):]
    prev_prices = df_test['Close'].shift(1).values[-len(predictions_std):]
    
    pred_dir = (predictions_std > prev_prices).astype(float)
    actual_dir = (actual_prices > prev_prices).astype(float)
    baseline_acc = (pred_dir == actual_dir).mean() * 100
    
    print(f"Baseline Accuracy: {baseline_acc:.2f}% (predicts all days)")
    
    # Test different confidence thresholds
    print(f"\n{'='*80}")
    print("TESTING CONFIDENCE THRESHOLDS")
    print(f"{'='*80}")
    
    results = evaluate_with_confidence_threshold(
        model, 
        df_test,
        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    
    # Analysis
    best = max(results, key=lambda x: x['selective_acc'])
    improvement = best['selective_acc'] - baseline_acc
    
    print(f"\n{'='*80}")
    print("📊 ANALYSIS")
    print(f"{'='*80}")
    print(f"\nBaseline (predict all): {baseline_acc:.2f}%")
    print(f"Best selective (threshold={best['threshold']:.2f}): {best['selective_acc']:.2f}%")
    print(f"Coverage: {best['coverage']:.1f}%")
    print(f"Improvement: {improvement:+.2f}%")
    
    if improvement > 5:
        print(f"\n✅ SUCCESS! Confidence-based abstention improves accuracy by {improvement:.1f}%")
    elif improvement > 0:
        print(f"\n✓ Modest improvement of {improvement:.1f}%")
    else:
        print(f"\n⚠ No improvement. Model uncertainty not well-calibrated.")
    
    return {
        'pattern': pattern_name,
        'baseline_acc': baseline_acc,
        'best_selective_acc': best['selective_acc'],
        'best_threshold': best['threshold'],
        'best_coverage': best['coverage'],
        'improvement': improvement
    }


def run_full_test():
    """
    Test confidence estimator across multiple patterns
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     SIMPLE CONFIDENCE ESTIMATOR TEST (MC Dropout)               ║
╚══════════════════════════════════════════════════════════════════╝

Testing whether dropout-based uncertainty estimation can provide
effective selective prediction WITHOUT model architecture changes.

This is simpler than the dual-head approach and requires NO retraining.
""")
    
    patterns = [
        ('Momentum', lambda: generate_momentum(n_days=400)),
        ('Mean Reversion', lambda: generate_mean_reversion(n_days=400)),
        ('Complex Pattern', lambda: generate_complex_pattern(n_days=400)),
    ]
    
    all_results = []
    
    for pattern_name, generator in patterns:
        df = generator()
        result = test_confidence_estimator(pattern_name, df)
        all_results.append(result)
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Pattern':<20} {'Baseline':<12} {'Selective':<12} {'Coverage':<12} {'Improvement'}")
    print("-" * 75)
    
    for r in all_results:
        print(f"{r['pattern']:<20} {r['baseline_acc']:>10.2f}% {r['best_selective_acc']:>10.2f}% "
              f"{r['best_coverage']:>10.1f}% {r['improvement']:>10.2f}%")
    
    avg_improvement = sum(r['improvement'] for r in all_results) / len(all_results)
    avg_coverage = sum(r['best_coverage'] for r in all_results) / len(all_results)
    
    print(f"\n{'Average':<20} {'':<12} {'':<12} {avg_coverage:>10.1f}% {avg_improvement:>10.2f}%")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")
    
    if avg_improvement > 5:
        print("✅ SUCCESS! Monte Carlo Dropout provides effective selective prediction!")
        print(f"   Average improvement: {avg_improvement:+.1f}% on {avg_coverage:.0f}% of days")
        print("\n💡 Recommendation: Use this simple approach")
        print("   - No model changes needed")
        print("   - Works with any trained LSTM")
        print("   - Just tune the confidence threshold")
    elif avg_improvement > 0:
        print("✓ Modest success. MC Dropout provides some benefit.")
        print(f"   Average improvement: {avg_improvement:+.1f}% on {avg_coverage:.0f}% of days")
        print("\n💡 Consider:")
        print("   - Tuning dropout rate")
        print("   - Increasing n_samples for MC Dropout")
        print("   - Or try the dual-head selective LSTM for better calibration")
    else:
        print("⚠ MC Dropout uncertainty not well-calibrated for this data.")
        print("\n💡 Next steps:")
        print("   - May need learned confidence (dual-head model)")
        print("   - Or different uncertainty quantification method")
        print("   - Check if dropout is actually active during training")


if __name__ == "__main__":
    run_full_test()
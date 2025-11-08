"""
Run LSTM tests using the optimal configuration from hyperparameter tuning
"""
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_lstm_synthetic_data import run_full_test_suite
from models.lstm_model import LSTMStockPredictor

def load_optimal_config(config_path: str = 'multi_pattern_results/best_config.json'):
    """
    Load the optimal configuration from tuning results
    
    Args:
        config_path: Path to best_config.json file
        
    Returns:
        Dictionary containing the config parameters
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    # Extract just the config parameters (not optimization_score, pattern_scores, etc.)
    config = data['config']
    
    print("="*80)
    print("LOADED OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"\nOptimization Score: {data.get('optimization_score', 'N/A')}")
    print(f"Average Score: {data.get('average_score', 'N/A')}%")
    print(f"Min Score: {data.get('min_score', 'N/A')}%")
    print(f"Max Score: {data.get('max_score', 'N/A')}%")
    
    if 'pattern_scores' in data:
        print(f"\nPattern Scores:")
        for pattern, score in data['pattern_scores'].items():
            print(f"  {pattern:20s}: {score:7.2f}%")
    
    print(f"\nConfiguration:")
    for key, val in config.items():
        # Skip model_dir and other non-essential params
        if key not in ['model_dir', 'epochs', 'batch_size', 'validation_sequences', 
                       'use_validation', 'early_stopping_patience', 'lr_patience', 
                       'lr_factor', 'min_lr', 'use_directional_loss', 'directional_loss_config']:
            print(f"  {key:25s}: {val}")
    
    print("="*80)
    print()
    
    return config


def run_with_optimal_config(config_path: str = 'multi_pattern_results/best_config.json'):
    """
    Run the full test suite with the optimal configuration
    
    Args:
        config_path: Path to best_config.json file
    """
    # Load config
    config = load_optimal_config(config_path)
    
    # Extract only the model parameters (not training parameters)
    model_params = {
        'sequence_length': config['sequence_length'],
        'hidden_size': config['hidden_size'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
        'learning_rate': config['learning_rate'],
        'bidirectional': config.get('bidirectional', True),
        'use_layer_norm': config.get('use_layer_norm', False),
        'use_residual': config.get('use_residual', False),
        'weight_decay': config.get('weight_decay', 1e-5),
        'gradient_clip_norm': config.get('gradient_clip_norm', 1.0),
    }
    
    print("Running full test suite with optimal configuration...")
    print()
    
    # Run tests
    results = run_full_test_suite(
        LSTMStockPredictor,
        **model_params
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests with optimal tuned configuration')
    parser.add_argument(
        '--config',
        type=str,
        default='multi_pattern_results/best_config.json',
        help='Path to best_config.json file (default: multi_pattern_results/best_config.json)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("\nAvailable options:")
        print("  1. Run multi-pattern tuning first: python tune_multi_pattern.py")
        print("  2. Or specify a different config path: python run_optimal_config.py --config path/to/best_config.json")
        sys.exit(1)
    
    # Run tests with optimal config
    run_with_optimal_config(args.config)
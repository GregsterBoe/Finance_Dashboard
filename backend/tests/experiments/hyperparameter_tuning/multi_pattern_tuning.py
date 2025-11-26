"""
Multi-Pattern LSTM Hyperparameter Tuning
=========================================

This module extends the base tuning to optimize across multiple market patterns,
finding configurations that work well across different scenarios.
"""

import sys
import os
# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
from typing import Dict, List
from tests.experiments.hyperparameter_tuning.lstm_hyperparameter_tuning import LSTMTuner, convert_to_serializable
from datetime import datetime
import json


class MultiPatternTuner:
    """
    Tune LSTM to maximize average performance across multiple patterns
    """
    
    def __init__(self, fixed_config_path: str, tunable_config_path: str,
                 results_dir: str = 'multi_pattern_tuning'):
        """
        Args:
            fixed_config_path: Path to fixed parameters JSON
            tunable_config_path: Path to tunable parameters JSON
            results_dir: Directory to save results
        """
        self.base_tuner = LSTMTuner(fixed_config_path, tunable_config_path, results_dir)
        self.results_dir = results_dir
        self.pattern_results = []
        self.best_config = None
        self.best_avg_score = float('-inf')
        
        os.makedirs(results_dir, exist_ok=True)
    
    def _evaluate_config_multi_pattern(self, config: Dict, 
                                       patterns_data: Dict[str, Dict],
                                       metric: str = 'direction_accuracy') -> Dict:
        """
        Evaluate configuration across multiple patterns
        
        Args:
            config: Configuration to test
            patterns_data: Dict of {pattern_name: {'train': df_train, 'val': df_val}}
            metric: Metric to optimize
        
        Returns:
            Dict with scores for each pattern and average
        """
        scores = {}
        
        for pattern_name, data in patterns_data.items():
            score = self.base_tuner._evaluate_config(
                config, 
                data['train'], 
                data['val'], 
                metric
            )
            scores[pattern_name] = float(score)
        
        # Calculate average score
        valid_scores = [s for s in scores.values() if s > float('-inf')]
        avg_score = np.mean(valid_scores) if valid_scores else float('-inf')
        
        return {
            'pattern_scores': scores,
            'average_score': float(avg_score),
            'min_score': float(min(valid_scores)) if valid_scores else float('-inf'),
            'max_score': float(max(valid_scores)) if valid_scores else float('-inf'),
            'std_score': float(np.std(valid_scores)) if valid_scores else 0.0
        }
    
    def random_search(self, patterns_data: Dict[str, Dict], 
                     n_iter: int = 50,
                     metric: str = 'direction_accuracy',
                     optimization_target: str = 'average') -> Dict:
        """
        Random search optimizing across multiple patterns
        
        Args:
            patterns_data: Dict of {pattern_name: {'train': df_train, 'val': df_val}}
            n_iter: Number of iterations
            metric: Metric to optimize
            optimization_target: 'average', 'min', or 'balanced'
                - 'average': Maximize average score across patterns
                - 'min': Maximize minimum score (worst case)
                - 'balanced': Maximize average - penalty for high variance
        
        Returns:
            Best configuration
        """
        from lstm_hyperparameter_tuning import RandomSearch
        
        print("="*80)
        print("MULTI-PATTERN RANDOM SEARCH")
        print("="*80)
        print(f"\nPatterns: {list(patterns_data.keys())}")
        print(f"Number of iterations: {n_iter}")
        print(f"Optimizing metric: {metric}")
        print(f"Optimization target: {optimization_target}")
        print("\nStarting search...\n")
        
        search = RandomSearch(self.base_tuner.tunable_params, n_iter)
        
        for i, tunable_config in enumerate(search, 1):
            full_config = self.base_tuner._merge_configs(tunable_config)
            
            print(f"[{i}/{n_iter}] Testing config:")
            for key, val in tunable_config.items():
                print(f"  {key}: {val}")
            
            # Evaluate on all patterns
            results = self._evaluate_config_multi_pattern(
                full_config, patterns_data, metric
            )
            
            print(f"\n  Pattern Scores:")
            for pattern, score in results['pattern_scores'].items():
                print(f"    {pattern:20s}: {score:7.2f}%")
            print(f"    {'─'*20}   {'─'*7}")
            print(f"    {'Average':20s}: {results['average_score']:7.2f}%")
            print(f"    {'Min (worst case)':20s}: {results['min_score']:7.2f}%")
            print(f"    {'Std Dev':20s}: {results['std_score']:7.2f}%")
            
            # Calculate optimization score based on target
            if optimization_target == 'average':
                opt_score = results['average_score']
            elif optimization_target == 'min':
                opt_score = results['min_score']
            elif optimization_target == 'balanced':
                # Penalize high variance - we want consistent performance
                opt_score = results['average_score'] - 0.5 * results['std_score']
            else:
                opt_score = results['average_score']
            
            print(f"    {'Optimization score':20s}: {opt_score:7.2f}%")
            
            # Store result
            result = {
                'config': convert_to_serializable(tunable_config),
                'full_config': convert_to_serializable(full_config),
                'pattern_scores': results['pattern_scores'],
                'average_score': results['average_score'],
                'min_score': results['min_score'],
                'max_score': results['max_score'],
                'std_score': results['std_score'],
                'optimization_score': float(opt_score),
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }
            
            self.pattern_results.append(result)
            
            # Check if this is best
            if opt_score > self.best_avg_score:
                self.best_avg_score = opt_score
                self.best_config = convert_to_serializable(full_config)
                print(f"\n  ✓ NEW BEST! Optimization score: {opt_score:.2f}%")
                self._save_best_config(results)
            
            print()
            
            # Save checkpoint every 10 iterations
            if i % 10 == 0 or i == n_iter:
                self._save_checkpoint()
        
        print("="*80)
        print("MULTI-PATTERN SEARCH COMPLETE")
        print("="*80)
        print(f"\nBest optimization score: {self.best_avg_score:.2f}%")
        print("\nBest configuration:")
        for key, val in self.best_config.items():
            print(f"  {key}: {val}")
        
        self._save_all_results()
        
        return self.best_config
    
    def grid_search(self, patterns_data: Dict[str, Dict],
                   metric: str = 'direction_accuracy',
                   optimization_target: str = 'average') -> Dict:
        """
        Grid search optimizing across multiple patterns
        
        Args:
            patterns_data: Dict of {pattern_name: {'train': df_train, 'val': df_val}}
            metric: Metric to optimize
            optimization_target: 'average', 'min', or 'balanced'
        
        Returns:
            Best configuration
        """
        from lstm_hyperparameter_tuning import GridSearch
        
        print("="*80)
        print("MULTI-PATTERN GRID SEARCH")
        print("="*80)
        print(f"\nPatterns: {list(patterns_data.keys())}")
        print(f"Optimization target: {optimization_target}")
        
        search = GridSearch(self.base_tuner.tunable_params)
        total_configs = len(search)
        
        print(f"Total configurations: {total_configs}")
        print("\nStarting search...\n")
        
        for i, tunable_config in enumerate(search, 1):
            full_config = self.base_tuner._merge_configs(tunable_config)
            
            print(f"[{i}/{total_configs}] Testing config:")
            for key, val in tunable_config.items():
                print(f"  {key}: {val}")
            
            results = self._evaluate_config_multi_pattern(
                full_config, patterns_data, metric
            )
            
            print(f"\n  Pattern Scores:")
            for pattern, score in results['pattern_scores'].items():
                print(f"    {pattern:20s}: {score:7.2f}%")
            print(f"    {'Average':20s}: {results['average_score']:7.2f}%")
            
            # Calculate optimization score
            if optimization_target == 'average':
                opt_score = results['average_score']
            elif optimization_target == 'min':
                opt_score = results['min_score']
            elif optimization_target == 'balanced':
                opt_score = results['average_score'] - 0.5 * results['std_score']
            else:
                opt_score = results['average_score']
            
            result = {
                'config': convert_to_serializable(tunable_config),
                'full_config': convert_to_serializable(full_config),
                'pattern_scores': results['pattern_scores'],
                'average_score': results['average_score'],
                'min_score': results['min_score'],
                'max_score': results['max_score'],
                'std_score': results['std_score'],
                'optimization_score': float(opt_score),
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }
            
            self.pattern_results.append(result)
            
            if opt_score > self.best_avg_score:
                self.best_avg_score = opt_score
                self.best_config = convert_to_serializable(full_config)
                print(f"  ✓ NEW BEST! Score: {opt_score:.2f}%")
                self._save_best_config(results)
            
            print()
            
            if i % 10 == 0 or i == total_configs:
                self._save_checkpoint()
        
        print("="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        
        self._save_all_results()
        return self.best_config
    
    def _save_best_config(self, results: Dict):
        """Save best configuration"""
        if self.best_config is None:
            return
        
        path = os.path.join(self.results_dir, 'best_config.json')
        data = {
            'optimization_score': float(self.best_avg_score),
            'config': convert_to_serializable(self.best_config),
            'pattern_scores': results['pattern_scores'],
            'average_score': results['average_score'],
            'min_score': results['min_score'],
            'max_score': results['max_score'],
            'std_score': results['std_score'],
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        path = os.path.join(self.results_dir, 'checkpoint.json')
        data = {
            'results': convert_to_serializable(self.pattern_results),
            'best_avg_score': float(self.best_avg_score),
            'best_config': convert_to_serializable(self.best_config),
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_all_results(self):
        """Save all results"""
        # Save JSON
        json_path = os.path.join(self.results_dir, 'all_results.json')
        data = {
            'results': convert_to_serializable(self.pattern_results),
            'best_avg_score': float(self.best_avg_score),
            'best_config': convert_to_serializable(self.best_config),
            'n_configs_tested': len(self.pattern_results),
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save CSV
        csv_path = os.path.join(self.results_dir, 'all_results.csv')
        
        # Flatten results for CSV
        rows = []
        for r in self.pattern_results:
            row = r['config'].copy()
            row['average_score'] = r['average_score']
            row['min_score'] = r['min_score']
            row['max_score'] = r['max_score']
            row['std_score'] = r['std_score']
            row['optimization_score'] = r['optimization_score']
            
            # Add individual pattern scores
            for pattern, score in r['pattern_scores'].items():
                row[f'score_{pattern}'] = score
            
            rows.append(row)
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")


# ============================================================================
# Helper Function to Generate All Patterns
# ============================================================================

def generate_all_patterns(n_days: int = 300, train_split: float = 0.8):
    """
    Generate all test patterns for tuning
    
    Args:
        n_days: Number of days per pattern
        train_split: Train/val split ratio
    
    Returns:
        Dict of {pattern_name: {'train': df_train, 'val': df_val}}
    """
    from test_lstm_synthetic_data import (
        generate_simple_trend,
        generate_mean_reversion,
        generate_momentum,
        generate_complex_pattern,
        generate_sideways_market,
        generate_volatile_market
    )
    
    patterns = {}
    
    # 1. Simple upward trend
    print("Generating simple trend pattern...")
    df = generate_simple_trend(n_days=n_days, trend=0.0005)  # Fixed: trend not trend_strength
    train_size = int(len(df) * train_split)
    patterns['trend'] = {
        'train': df.iloc[:train_size],
        'val': df.iloc[train_size:]
    }
    
    # 2. Mean reversion
    print("Generating mean reversion pattern...")
    df = generate_mean_reversion(n_days=n_days, reversion_speed=0.3)  # Fixed: reversion_speed not reversion_strength
    train_size = int(len(df) * train_split)
    patterns['mean_reversion'] = {
        'train': df.iloc[:train_size],
        'val': df.iloc[train_size:]
    }
    
    # 3. Momentum
    print("Generating momentum pattern...")
    df = generate_momentum(n_days=n_days, momentum_strength=0.6)  # This one is correct
    train_size = int(len(df) * train_split)
    patterns['momentum'] = {
        'train': df.iloc[:train_size],
        'val': df.iloc[train_size:]
    }
    
    # 4. Complex pattern
    print("Generating complex pattern...")
    df = generate_complex_pattern(n_days=n_days)
    train_size = int(len(df) * train_split)
    patterns['complex'] = {
        'train': df.iloc[:train_size],
        'val': df.iloc[train_size:]
    }
    
    # 5. Sideways market
    print("Generating sideways market...")
    df = generate_sideways_market(n_days=n_days, range_pct=0.05)
    train_size = int(len(df) * train_split)
    patterns['sideways'] = {
        'train': df.iloc[:train_size],
        'val': df.iloc[train_size:]
    }
    
    # 6. Volatile market
    print("Generating volatile market...")
    df = generate_volatile_market(n_days=n_days, volatility=0.03)
    train_size = int(len(df) * train_split)
    patterns['volatile'] = {
        'train': df.iloc[:train_size],
        'val': df.iloc[train_size:]
    }
    
    print(f"\nGenerated {len(patterns)} patterns")
    return patterns


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     MULTI-PATTERN LSTM HYPERPARAMETER TUNING                     ║
╚══════════════════════════════════════════════════════════════════╝

This will optimize LSTM configuration across multiple market patterns
to find a robust configuration that works well in all scenarios.

Patterns tested:
  1. Simple Trend
  2. Mean Reversion
  3. Momentum
  4. Complex Pattern
  5. Sideways Market
  6. Volatile Market

Optimization targets:
  - 'average': Maximize average score across all patterns
  - 'min': Maximize worst-case performance
  - 'balanced': Maximize average with penalty for inconsistency
""")
    
    # Generate all patterns
    print("\nGenerating test patterns...")
    patterns_data = generate_all_patterns(n_days=300)
    
    # Create tuner
    print("\nCreating multi-pattern tuner...")
    tuner = MultiPatternTuner(
        fixed_config_path='fixed_config.json',
        tunable_config_path='tunable_config_quick.json',
        results_dir='multi_pattern_results'
    )
    
    # Run search
    print("\nStarting optimization...")
    best_config = tuner.random_search(
        patterns_data=patterns_data,
        n_iter=20,
        metric='direction_accuracy',
        optimization_target='average'  # or 'min' or 'balanced'
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nBest average score: {tuner.best_avg_score:.2f}%")
    print("\nUse this config for robust performance across all market conditions!")
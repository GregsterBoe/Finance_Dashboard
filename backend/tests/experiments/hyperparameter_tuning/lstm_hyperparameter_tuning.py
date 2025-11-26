"""
LSTM Hyperparameter Tuning Framework
=====================================

This module provides a complete hyperparameter optimization framework for LSTM models.

Features:
- Grid Search: Exhaustive search over parameter space
- Random Search: Sample random combinations (faster)
- Bayesian Optimization: Smart search using previous results
- Config-driven: All parameters in JSON files
- Resume capability: Continue interrupted searches
- Cross-validation: K-fold validation for robust results
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os
import sys
import itertools
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add backend directory to path to allow imports from models/
# Go up 3 levels: hyperparameter_tuning -> experiments -> tests -> backend
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


# ============================================================================
# Helper Functions
# ============================================================================

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

# ============================================================================
# Configuration Classes
# ============================================================================

class HyperparameterConfig:
    """Configuration for a single hyperparameter to tune"""
    
    def __init__(self, name: str, min_val: float, max_val: float, 
                 step: Optional[float] = None, type: str = 'int',
                 scale: str = 'linear', values: Optional[List] = None):
        """
        Args:
            name: Parameter name
            min_val: Minimum value
            max_val: Maximum value
            step: Step size (for grid search)
            type: 'int', 'float', or 'categorical'
            scale: 'linear' or 'log' (for random/bayesian search)
            values: Explicit list of values to try (for categorical)
        """
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
        else:  # float
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
            # Sample in log space
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            value = 10 ** np.random.uniform(log_min, log_max)
        else:
            # Linear sampling
            value = np.random.uniform(self.min_val, self.max_val)
        
        if self.type == 'int':
            return int(round(value))
        else:
            return float(value)


# ============================================================================
# Tuning Strategies
# ============================================================================

class GridSearch:
    """Exhaustive grid search over parameter space"""
    
    def __init__(self, tunable_params: Dict[str, HyperparameterConfig]):
        self.tunable_params = tunable_params
        self.param_grid = self._build_grid()
    
    def _build_grid(self) -> List[Dict]:
        """Build complete parameter grid"""
        param_names = list(self.tunable_params.keys())
        param_values = [self.tunable_params[name].get_grid_values() 
                       for name in param_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dicts
        grid = []
        for combo in combinations:
            config = {name: value for name, value in zip(param_names, combo)}
            grid.append(config)
        
        return grid
    
    def __len__(self):
        return len(self.param_grid)
    
    def __iter__(self):
        return iter(self.param_grid)


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


class BayesianOptimization:
    """Bayesian optimization using Gaussian Process (requires scikit-optimize)"""
    
    def __init__(self, tunable_params: Dict[str, HyperparameterConfig],
                 n_iter: int = 50, n_initial_points: int = 10):
        self.tunable_params = tunable_params
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.history = []
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            self.has_skopt = True
        except ImportError:
            print("WARNING: scikit-optimize not installed. Falling back to RandomSearch.")
            print("Install with: pip install scikit-optimize")
            self.has_skopt = False
    
    def _build_search_space(self):
        """Build search space for skopt"""
        from skopt.space import Real, Integer, Categorical
        
        space = []
        param_names = []
        
        for name, param in self.tunable_params.items():
            param_names.append(name)
            
            if param.values is not None:
                space.append(Categorical(param.values, name=name))
            elif param.type == 'int':
                space.append(Integer(int(param.min_val), int(param.max_val), name=name))
            else:
                if param.scale == 'log':
                    space.append(Real(param.min_val, param.max_val, prior='log-uniform', name=name))
                else:
                    space.append(Real(param.min_val, param.max_val, name=name))
        
        return space, param_names
    
    def optimize(self, objective_func):
        """Run Bayesian optimization"""
        if not self.has_skopt:
            # Fallback to random search
            return RandomSearch(self.tunable_params, self.n_iter)
        
        from skopt import gp_minimize
        
        space, param_names = self._build_search_space()
        
        def objective_wrapper(params):
            config = {name: val for name, val in zip(param_names, params)}
            score = objective_func(config)
            self.history.append((config, score))
            return -score  # Minimize negative score (maximize score)
        
        result = gp_minimize(
            objective_wrapper,
            space,
            n_calls=self.n_iter,
            n_initial_points=self.n_initial_points,
            random_state=42
        )
        
        return result


# ============================================================================
# LSTM Tuner
# ============================================================================

class LSTMTuner:
    """Main tuning class for LSTM models"""
    
    def __init__(self, fixed_config_path: str, tunable_config_path: str,
                 results_dir: str = 'tuning_results'):
        """
        Args:
            fixed_config_path: Path to fixed parameters JSON
            tunable_config_path: Path to tunable parameters JSON
            results_dir: Directory to save results
        """
        # Initialize instance variables FIRST (before calling methods)
        self.results_dir = results_dir
        self.results = []
        self.best_config = None
        self.best_score = float('-inf')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load configurations (these methods depend on instance variables)
        self.fixed_params = self._load_fixed_config(fixed_config_path)
        self.tunable_params = self._load_tunable_config(tunable_config_path)
        
        print(f"Loaded {len(self.tunable_params)} tunable parameters")
        print(f"Results will be saved to: {results_dir}")
    
    def _load_fixed_config(self, path: str) -> Dict:
        """Load fixed parameters from JSON"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_tunable_config(self, path: str) -> Dict[str, HyperparameterConfig]:
        """Load tunable parameters from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        tunable_params = {}
        for name, spec in config_dict.items():
            # Skip internal keys (starting with _)
            if name.startswith('_'):
                continue
            
            # Skip if spec is not a dict (e.g., string descriptions)
            if not isinstance(spec, dict):
                continue
            
            # Check if this uses explicit values list or min/max range
            if 'values' in spec:
                # Explicit values provided
                tunable_params[name] = HyperparameterConfig(
                    name=name,
                    min_val=0,  # Not used when values provided
                    max_val=1,  # Not used when values provided
                    step=None,
                    type=spec.get('type', 'categorical'),
                    scale=spec.get('scale', 'linear'),
                    values=spec['values']
                )
            else:
                # Min/max range provided
                tunable_params[name] = HyperparameterConfig(
                    name=name,
                    min_val=spec['min'],
                    max_val=spec['max'],
                    step=spec.get('step'),
                    type=spec.get('type', 'int'),
                    scale=spec.get('scale', 'linear'),
                    values=None
                )
        
        return tunable_params
    
    def _merge_configs(self, tunable_values: Dict) -> Dict:
        """Merge fixed and tunable parameters"""
        config = {}
        
        # Add fixed parameters (skip internal keys)
        for key, val in self.fixed_params.items():
            if not key.startswith('_'):
                config[key] = val
        
        # Override with tunable values
        config.update(tunable_values)
        
        return config
    
    def _evaluate_config(self, config: Dict, df_train: pd.DataFrame, 
                        df_val: pd.DataFrame, metric: str = 'direction_accuracy') -> float:
        """
        Evaluate a single configuration
        
        Args:
            config: Parameter configuration
            df_train: Training data
            df_val: Validation data
            metric: Metric to optimize ('direction_accuracy', 'rmse', 'r2', etc.)
        
        Returns:
            Score (higher is better)
        """
        # Try different import paths depending on where script is run from
        try:
            from models.lstm_model import LSTMStockPredictor
        except ModuleNotFoundError:
            # Import is already handled at top of file
            from models.lstm_model import LSTMStockPredictor
        
        try:
            # Separate model parameters from training parameters
            model_params = [
                'sequence_length', 'hidden_size', 'num_layers', 'dropout',
                'learning_rate', 'model_dir', 'bidirectional', 'use_layer_norm',
                'use_residual', 'weight_decay', 'gradient_clip_norm',
                'use_directional_loss', 'directional_loss_config'
            ]
            
            training_params = [
                'epochs', 'batch_size', 'validation_sequences', 'use_validation',
                'early_stopping_patience', 'lr_patience', 'lr_factor', 'min_lr'
            ]
            
            # Extract model parameters
            model_config = {k: v for k, v in config.items() if k in model_params}
            
            # Extract training parameters
            train_config = {k: v for k, v in config.items() if k in training_params}
            
            # Convert all numpy types to Python types before passing to model
            model_config = convert_to_serializable(model_config)
            train_config = convert_to_serializable(train_config)
            
            # Create predictor with model params only
            predictor = LSTMStockPredictor(**model_config)
            
            # Train with training params
            history = predictor.train(
                df_train,
                **train_config
            )
            
            # Evaluate on validation set
            val_metrics = predictor.get_validation_metrics()
            
            # Get score based on metric
            if metric == 'direction_accuracy':
                score = val_metrics['direction_accuracy']
            elif metric == 'rmse':
                score = -val_metrics['rmse']  # Negative because lower is better
            elif metric == 'r2':
                score = val_metrics['r2_score']
            elif metric == 'mae':
                score = -val_metrics['mae']  # Negative because lower is better
            else:
                score = val_metrics['direction_accuracy']
            
            return float(score)  # Convert to Python float
        
        except Exception as e:
            print(f"Error evaluating config: {e}")
            return float('-inf')
    
    def grid_search(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                   metric: str = 'direction_accuracy',
                   save_all: bool = True) -> Dict:
        """
        Perform grid search
        
        Args:
            df_train: Training data
            df_val: Validation data
            metric: Metric to optimize
            save_all: Whether to save all results or just best
        
        Returns:
            Best configuration
        """
        print("="*80)
        print("GRID SEARCH")
        print("="*80)
        
        search = GridSearch(self.tunable_params)
        total_configs = len(search)
        
        print(f"\nTotal configurations to try: {total_configs}")
        print(f"Optimizing metric: {metric}")
        print("\nStarting search...\n")
        
        for i, tunable_config in enumerate(search, 1):
            full_config = self._merge_configs(tunable_config)
            
            print(f"[{i}/{total_configs}] Testing config:")
            for key, val in tunable_config.items():
                print(f"  {key}: {val}")
            
            score = self._evaluate_config(full_config, df_train, df_val, metric)
            
            print(f"  → Score: {score:.4f}")
            
            result = {
                'config': convert_to_serializable(tunable_config),
                'full_config': convert_to_serializable(full_config),
                'score': float(score),
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = full_config
                print(f"  ✓ NEW BEST! Score: {score:.4f}")
                
                # Save best config immediately
                self._save_best_config()
            
            print()
            
            # Save checkpoint every 10 configs
            if i % 10 == 0 or i == total_configs:
                self._save_checkpoint()
        
        print("="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        print(f"\nBest score: {self.best_score:.4f}")
        print("Best configuration:")
        for key, val in self.best_config.items():
            print(f"  {key}: {val}")
        
        # Save final results
        self._save_all_results()
        
        return self.best_config
    
    def random_search(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                     n_iter: int = 50, metric: str = 'direction_accuracy') -> Dict:
        """
        Perform random search
        
        Args:
            df_train: Training data
            df_val: Validation data
            n_iter: Number of iterations
            metric: Metric to optimize
        
        Returns:
            Best configuration
        """
        print("="*80)
        print("RANDOM SEARCH")
        print("="*80)
        
        search = RandomSearch(self.tunable_params, n_iter)
        
        print(f"\nNumber of iterations: {n_iter}")
        print(f"Optimizing metric: {metric}")
        print("\nStarting search...\n")
        
        for i, tunable_config in enumerate(search, 1):
            full_config = self._merge_configs(tunable_config)
            
            print(f"[{i}/{n_iter}] Testing config:")
            for key, val in tunable_config.items():
                print(f"  {key}: {val}")
            
            score = self._evaluate_config(full_config, df_train, df_val, metric)
            
            print(f"  → Score: {score:.4f}")
            
            result = {
                'config': convert_to_serializable(tunable_config),
                'full_config': convert_to_serializable(full_config),
                'score': float(score),
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = convert_to_serializable(full_config)
                print(f"  ✓ NEW BEST! Score: {score:.4f}")
                self._save_best_config()
            
            print()
            
            # Save checkpoint every 10 iterations
            if i % 10 == 0 or i == n_iter:
                self._save_checkpoint()
        
        print("="*80)
        print("RANDOM SEARCH COMPLETE")
        print("="*80)
        print(f"\nBest score: {self.best_score:.4f}")
        print("Best configuration:")
        for key, val in self.best_config.items():
            print(f"  {key}: {val}")
        
        self._save_all_results()
        
        return self.best_config
    
    def bayesian_search(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                       n_iter: int = 50, metric: str = 'direction_accuracy') -> Dict:
        """
        Perform Bayesian optimization
        
        Args:
            df_train: Training data
            df_val: Validation data
            n_iter: Number of iterations
            metric: Metric to optimize
        
        Returns:
            Best configuration
        """
        print("="*80)
        print("BAYESIAN OPTIMIZATION")
        print("="*80)
        
        print(f"\nNumber of iterations: {n_iter}")
        print(f"Optimizing metric: {metric}")
        print("\nStarting search...\n")
        
        optimizer = BayesianOptimization(self.tunable_params, n_iter)
        
        if not optimizer.has_skopt:
            print("Falling back to random search...")
            return self.random_search(df_train, df_val, n_iter, metric)
        
        iteration = [0]  # Use list to modify in nested function
        
        def objective(tunable_config):
            iteration[0] += 1
            full_config = self._merge_configs(tunable_config)
            
            print(f"[{iteration[0]}/{n_iter}] Testing config:")
            for key, val in tunable_config.items():
                print(f"  {key}: {val}")
            
            score = self._evaluate_config(full_config, df_train, df_val, metric)
            
            print(f"  → Score: {score:.4f}")
            
            result = {
                'config': convert_to_serializable(tunable_config),
                'full_config': convert_to_serializable(full_config),
                'score': float(score),
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = convert_to_serializable(full_config)
                print(f"  ✓ NEW BEST! Score: {score:.4f}")
                self._save_best_config()
            
            print()
            
            if iteration[0] % 10 == 0:
                self._save_checkpoint()
            
            return score
        
        # Run optimization
        optimizer.optimize(objective)
        
        print("="*80)
        print("BAYESIAN OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nBest score: {self.best_score:.4f}")
        print("Best configuration:")
        for key, val in self.best_config.items():
            print(f"  {key}: {val}")
        
        self._save_all_results()
        
        return self.best_config
    
    def _save_best_config(self):
        """Save best configuration to file"""
        if self.best_config is None:
            return  # Nothing to save yet
        
        path = os.path.join(self.results_dir, 'best_config.json')
        data = {
            'score': float(self.best_score),
            'config': convert_to_serializable(self.best_config),
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_checkpoint(self):
        """Save checkpoint of all results so far"""
        path = os.path.join(self.results_dir, 'checkpoint.json')
        data = {
            'results': convert_to_serializable(self.results),
            'best_score': float(self.best_score),
            'best_config': convert_to_serializable(self.best_config),
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_all_results(self):
        """Save all results to file"""
        # Save as JSON
        json_path = os.path.join(self.results_dir, 'all_results.json')
        data = {
            'results': convert_to_serializable(self.results),
            'best_score': float(self.best_score),
            'best_config': convert_to_serializable(self.best_config),
            'n_configs_tested': len(self.results),
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_path = os.path.join(self.results_dir, 'all_results.csv')
        df_results = pd.DataFrame([
            {**convert_to_serializable(r['config']), 'score': float(r['score'])} 
            for r in self.results
        ])
        df_results.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume search"""
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        self.results = data['results']
        self.best_score = data['best_score']
        self.best_config = data['best_config']
        
        print(f"Loaded checkpoint with {len(self.results)} results")
        print(f"Best score so far: {self.best_score:.4f}")


# ============================================================================
# Visualization and Analysis
# ============================================================================

def analyze_results(results_path: str):
    """Analyze tuning results and generate visualizations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load results
    df = pd.read_csv(results_path)
    
    print("="*80)
    print("HYPERPARAMETER TUNING ANALYSIS")
    print("="*80)
    
    # Summary statistics
    print("\nScore Statistics:")
    print(f"  Best score: {df['score'].max():.4f}")
    print(f"  Mean score: {df['score'].mean():.4f}")
    print(f"  Std score: {df['score'].std():.4f}")
    print(f"  Worst score: {df['score'].min():.4f}")
    
    # Top 10 configurations
    print("\nTop 10 Configurations:")
    top_10 = df.nlargest(10, 'score')
    print(top_10.to_string(index=False))
    
    # Parameter importance (correlation with score)
    print("\nParameter Importance (correlation with score):")
    param_cols = [col for col in df.columns if col != 'score']
    correlations = {}
    for col in param_cols:
        if df[col].dtype in [np.int64, np.float64]:
            corr = df[col].corr(df['score'])
            correlations[col] = abs(corr)
    
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for param, corr in sorted_corr:
        print(f"  {param:25s}: {corr:.4f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Score distribution
    axes[0, 0].hist(df['score'], bins=30, edgecolor='black')
    axes[0, 0].axvline(df['score'].max(), color='r', linestyle='--', label='Best')
    axes[0, 0].axvline(df['score'].mean(), color='g', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Score over time
    axes[0, 1].plot(df['score'], alpha=0.6)
    axes[0, 1].plot(df['score'].cummax(), 'r-', linewidth=2, label='Best so far')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Score Over Iterations')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Parameter importance bar chart
    top_params = sorted_corr[:min(10, len(sorted_corr))]
    params, corrs = zip(*top_params)
    axes[1, 0].barh(params, corrs)
    axes[1, 0].set_xlabel('Absolute Correlation with Score')
    axes[1, 0].set_title('Top Parameter Importance')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Scatter plot of two most important parameters
    if len(sorted_corr) >= 2:
        param1, _ = sorted_corr[0]
        param2, _ = sorted_corr[1]
        scatter = axes[1, 1].scatter(df[param1], df[param2], c=df['score'], 
                                     cmap='viridis', s=50, alpha=0.6)
        axes[1, 1].set_xlabel(param1)
        axes[1, 1].set_ylabel(param2)
        axes[1, 1].set_title(f'{param1} vs {param2} (colored by score)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Score')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_path.replace('.csv', '_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {plot_path}")
    
    return df


if __name__ == "__main__":
    print("""
    LSTM Hyperparameter Tuning Framework
    ====================================
    
    This module provides tools for tuning LSTM hyperparameters.
    
    See example_config files for usage.
    """)
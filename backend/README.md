# LSTM Hyperparameter Tuning - Quick Command Reference

## Setup

```bash
cd backend/test
```

Ensure you have the configuration files:
- `fixed_config.json` - Fixed training parameters
- `tunable_config_quick.json` - Parameter search space

## Quick Start Commands

### Single-Pattern Tuning

**Random Search** (recommended for quick results):
```bash
python tune_lstm.py
```
Tests 20 random configurations on synthetic momentum data.

### Multi-Pattern Tuning

**Robust Search** (tests across 6 market patterns):
```bash
python tune_multi_pattern.py
```
Finds configuration that works well across diverse market conditions.

## Python API Usage

### Single-Pattern Random Search

```python
from lstm_hyperparameter_tuning import LSTMTuner
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:]

# Create tuner
tuner = LSTMTuner(
    fixed_config_path='fixed_config.json',
    tunable_config_path='tunable_config_quick.json',
    results_dir='tuning_results'
)

# Run search
best_config = tuner.random_search(
    df_train=df_train,
    df_val=df_val,
    n_iter=20,  # Number of configs to try
    metric='direction_accuracy'
)
```

### Single-Pattern Grid Search

```python
# Tests ALL parameter combinations (~324 configs)
best_config = tuner.grid_search(
    df_train=df_train,
    df_val=df_val,
    metric='direction_accuracy'
)
```

### Multi-Pattern Random Search

```python
from multi_pattern_tuning import MultiPatternTuner, generate_all_patterns

# Generate test patterns
patterns = generate_all_patterns(n_days=300)

# Create tuner
tuner = MultiPatternTuner(
    fixed_config_path='fixed_config.json',
    tunable_config_path='tunable_config_quick.json',
    results_dir='multi_pattern_results'
)

# Run search
best_config = tuner.random_search(
    patterns_data=patterns,
    n_iter=20,
    metric='direction_accuracy',
    optimization_target='average'  # 'average', 'min', or 'balanced'
)
```

### Multi-Pattern Grid Search

```python
best_config = tuner.grid_search(
    patterns_data=patterns,
    metric='direction_accuracy',
    optimization_target='average'
)
```

## Available Metrics

- `'direction_accuracy'` - Correct price direction predictions (default)
- `'rmse'` - Root Mean Square Error
- `'r2'` - R-squared score
- `'mae'` - Mean Absolute Error

## Optimization Targets (Multi-Pattern Only)

- `'average'` - Maximize average score across patterns (default)
- `'min'` - Maximize worst-case performance
- `'balanced'` - Average score with variance penalty

## Output Files

Results are saved to the specified results directory:

- `best_config.json` - Best configuration found
- `all_results.json` - All tested configurations
- `all_results.csv` - Results in CSV format
- `checkpoint.json` - Progress checkpoint (every 10 iterations)

## Typical Runtime (tunable_config_quick.json)

- Random search (20 iter): ~10-40 minutes
- Grid search (324 configs): ~3-11 hours
- Multi-pattern (6 patterns × 20 iter): ~1-2 hours

## Using Results

### Run Tests with Optimal Config

After tuning, use the optimal configuration for testing:

```bash
python run_optimal_config.py
```

By default, this loads `multi_pattern_results/best_config.json`. To use a different config:

```bash
python run_optimal_config.py --config tuning_results/best_config.json
```

This script:
1. Loads the best configuration from tuning
2. Displays the optimization scores and pattern performance
3. Runs the full test suite with the optimal parameters

### Load Config in Python

```python
import json

# Load best config
with open('multi_pattern_results/best_config.json', 'r') as f:
    data = json.load(f)
    best_config = data['config']

# Use in your model
from models.lstm_model import LSTMStockPredictor

model = LSTMStockPredictor(
    sequence_length=best_config['sequence_length'],
    hidden_size=best_config['hidden_size'],
    num_layers=best_config['num_layers'],
    dropout=best_config['dropout'],
    learning_rate=best_config['learning_rate'],
    bidirectional=best_config['bidirectional'],
    use_layer_norm=best_config['use_layer_norm'],
    use_residual=best_config['use_residual'],
    weight_decay=best_config['weight_decay'],
    gradient_clip_norm=best_config['gradient_clip_norm']
)
```
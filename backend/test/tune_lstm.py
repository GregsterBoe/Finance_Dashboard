import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_hyperparameter_tuning import LSTMTuner
from test_lstm_synthetic_data import generate_momentum

# 1. Generate test data
df = generate_momentum(n_days=400)
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:]

# 2. Create tuner
tuner = LSTMTuner(
    fixed_config_path='fixed_config.json',
    tunable_config_path='tunable_config_quick.json',
    results_dir='tuning_results'
)

# 3. Run search
best_config = tuner.random_search(
    df_train=df_train,
    df_val=df_val,
    n_iter=20,  # Try 20 random configs
    metric='direction_accuracy'
)

# 4. View results
print(f"Best score: {tuner.best_score:.2f}%")
print(f"Best config: {best_config}")
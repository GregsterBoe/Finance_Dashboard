import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_lstm_synthetic_data import run_full_test_suite
from models.lstm_model import LSTMStockPredictor

BASELINE_CONFIG = {
    'sequence_length': 5,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'bidirectional': True,        # ← MAIN ISSUE!
    'use_layer_norm': True,       # ← Too much regularization
    'use_residual': True,         # ← Too much regularization
    'weight_decay': 0.01,         # ← Default
    'gradient_clip_norm': 1.0
}

TUNED_CONFIG = {
  "sequence_length": 5,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.002,
    "bidirectional": True,
    "use_layer_norm": False,
    "use_residual": False,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0
}

OPTIMAL_CONFIG = {
    'sequence_length': 40,          # Longer memory
    'hidden_size': 128,             # More capacity
    'num_layers': 2,                # Keep at 2
    'dropout': 0.0,                 # Reduced regularization
    'learning_rate': 0.001,         # Keep same
    'bidirectional': True,         # ← CRITICAL: No bidirectional for time series
    'use_layer_norm': False,        # ← Simpler model
    'use_residual': False,          # ← Simpler model
    'weight_decay': 0.0001,         # ← Less L2 penalty
    'gradient_clip_norm': 1.0       # Keep same
}

results = run_full_test_suite(
    LSTMStockPredictor,
    **TUNED_CONFIG
)
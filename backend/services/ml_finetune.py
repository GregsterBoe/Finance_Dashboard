"""
backend/services/ml_finetune.py

LSTM hyperparameter finetuning service with SSE progress streaming.
Uses random search over a configurable parameter space, evaluating each
configuration by training on real stock data and measuring validation metrics.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
import numpy as np
import torch

from models.lstm_model import LSTMStockPredictor
from services.data_provider import get_data_provider
from utils.hyperparameter_utils import (
    HyperparameterConfig,
    RandomSearch,
    convert_to_serializable,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FinetuneConfig(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    n_iterations: int = 15
    metric: str = "direction_accuracy"  # direction_accuracy | rmse | r2 | mae
    # Fixed training params (reduced epochs for speed during search)
    epochs: int = 50
    batch_size: int = 32
    validation_sequences: int = 30
    early_stopping_patience: int = 10
    use_validation: bool = True
    lr_patience: int = 7
    lr_factor: float = 0.5
    min_lr: float = 1e-6


# ---------------------------------------------------------------------------
# Default tunable parameter space (matches tunable_config.json)
# ---------------------------------------------------------------------------

def get_default_tunable_params() -> Dict[str, HyperparameterConfig]:
    return {
        "sequence_length": HyperparameterConfig(
            "sequence_length", 10, 60, step=10, type="int"),
        "hidden_size": HyperparameterConfig(
            "hidden_size", 32, 256, step=32, type="int"),
        "num_layers": HyperparameterConfig(
            "num_layers", 1, 3, step=1, type="int"),
        "dropout": HyperparameterConfig(
            "dropout", 0.0, 0.3, step=0.05, type="float"),
        "learning_rate": HyperparameterConfig(
            "learning_rate", 0.0001, 0.01, type="float", scale="log"),
        "bidirectional": HyperparameterConfig(
            "bidirectional", 0, 1, type="categorical", values=[False, True]),
        "use_layer_norm": HyperparameterConfig(
            "use_layer_norm", 0, 1, type="categorical", values=[False, True]),
        "use_residual": HyperparameterConfig(
            "use_residual", 0, 1, type="categorical", values=[False, True]),
        "weight_decay": HyperparameterConfig(
            "weight_decay", 0.00001, 0.01, type="float", scale="log"),
        "gradient_clip_norm": HyperparameterConfig(
            "gradient_clip_norm", 0.5, 5.0, step=0.5, type="float"),
    }


# ---------------------------------------------------------------------------
# Core evaluation function (adapted from LSTMTuner._evaluate_config)
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    'sequence_length', 'hidden_size', 'num_layers', 'dropout',
    'learning_rate', 'model_dir', 'bidirectional', 'use_layer_norm',
    'use_residual', 'weight_decay', 'gradient_clip_norm',
    'use_directional_loss', 'directional_loss_config',
}

TRAINING_PARAMS = {
    'epochs', 'batch_size', 'validation_sequences', 'use_validation',
    'early_stopping_patience', 'lr_patience', 'lr_factor', 'min_lr',
}


def evaluate_single_config(
    config: Dict[str, Any],
    df_train,
    df_val,
    metric: str = "direction_accuracy",
) -> float:
    """
    Train an LSTMStockPredictor with *config* on *df_train* and return a
    scalar score measured on *df_val*.  Higher is always better (RMSE / MAE
    are negated).
    """
    model_config = {k: v for k, v in config.items() if k in MODEL_PARAMS}
    train_config = {k: v for k, v in config.items() if k in TRAINING_PARAMS}

    # Ensure Python-native types (no numpy scalars)
    model_config = convert_to_serializable(model_config)
    train_config = convert_to_serializable(train_config)

    try:
        predictor = LSTMStockPredictor(**model_config)
        predictor.train(df_train, **train_config)
        val_metrics = predictor.get_validation_metrics()

        if metric == "direction_accuracy":
            score = val_metrics["direction_accuracy"]
        elif metric == "rmse":
            score = -val_metrics["rmse"]
        elif metric == "r2":
            score = val_metrics["r2_score"]
        elif metric == "mae":
            score = -val_metrics["mae"]
        else:
            score = val_metrics["direction_accuracy"]

        return float(score)
    except Exception as e:
        print(f"[finetune] Error evaluating config: {e}")
        return float("-inf")
    finally:
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _yield_progress(message: str, progress: float, data: dict = None):
    event_data = {"message": message, "progress": progress, **(data or {})}
    return f"data: {json.dumps(event_data)}\n\n"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/finetune-lstm-stream")
async def finetune_lstm_stream(config: FinetuneConfig):
    """Run a random hyperparameter search with SSE progress streaming."""

    async def event_generator():
        try:
            yield _yield_progress("Downloading stock data...", 0)
            await asyncio.sleep(0.05)

            # --- fetch data ---------------------------------------------------
            start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
            data_provider = get_data_provider()
            df = data_provider.get_stock_history(
                config.ticker, start=start_date, end=end_date + timedelta(days=5)
            )

            if df.empty or len(df) < 100:
                yield _yield_progress(
                    "Error: Need at least 100 trading days of data for finetuning.",
                    100, {"error": True},
                )
                return

            yield _yield_progress(
                f"Loaded {len(df)} trading days for {config.ticker}", 5
            )

            # --- chronological train / val split ------------------------------
            split_idx = int(len(df) * 0.8)
            df_train = df.iloc[:split_idx].copy()
            df_val = df.iloc[split_idx:].copy()

            yield _yield_progress(
                f"Split: {len(df_train)} train / {len(df_val)} validation days", 8
            )

            # --- build fixed params from request config -----------------------
            fixed_params = {
                "model_dir": "saved_models/finetune",
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "validation_sequences": config.validation_sequences,
                "use_validation": config.use_validation,
                "early_stopping_patience": config.early_stopping_patience,
                "lr_patience": config.lr_patience,
                "lr_factor": config.lr_factor,
                "min_lr": config.min_lr,
                "use_directional_loss": False,
            }

            # --- random search ------------------------------------------------
            tunable_params = get_default_tunable_params()
            search = RandomSearch(tunable_params, n_iter=config.n_iterations)

            best_score = float("-inf")
            best_config: Dict[str, Any] = {}
            all_results: List[Dict[str, Any]] = []

            yield _yield_progress(
                f"Starting random search ({config.n_iterations} iterations)...", 10
            )
            await asyncio.sleep(0.05)

            for i, sampled in enumerate(search, 1):
                # Merge fixed + sampled
                merged = {**fixed_params, **sampled}

                progress = 10 + (85 * i / config.n_iterations)

                yield _yield_progress(
                    f"Testing config {i}/{config.n_iterations}...",
                    progress,
                    {
                        "iteration": i,
                        "total_iterations": config.n_iterations,
                        "current_config": convert_to_serializable(sampled),
                    },
                )
                await asyncio.sleep(0.05)

                score = evaluate_single_config(
                    merged, df_train, df_val, config.metric
                )

                is_best = score > best_score
                if is_best:
                    best_score = score
                    best_config = convert_to_serializable(merged)

                result_entry = {
                    "iteration": i,
                    "config": convert_to_serializable(sampled),
                    "score": float(score) if score != float("-inf") else None,
                    "metric": config.metric,
                    "is_best": is_best,
                }
                all_results.append(result_entry)

                yield _yield_progress(
                    f"Config {i}/{config.n_iterations} — score: {score:.4f}"
                    + (" (new best!)" if is_best else ""),
                    progress,
                    {
                        "iteration": i,
                        "total_iterations": config.n_iterations,
                        "current_score": float(score) if score != float("-inf") else None,
                        "best_score": float(best_score) if best_score != float("-inf") else None,
                        "best_config": best_config,
                    },
                )
                await asyncio.sleep(0.05)

            # --- done ---------------------------------------------------------
            response = {
                "status": "completed",
                "ticker": config.ticker,
                "best_config": best_config,
                "best_score": float(best_score) if best_score != float("-inf") else None,
                "metric": config.metric,
                "all_results": all_results,
                "total_iterations": config.n_iterations,
                "timestamp": datetime.now().isoformat(),
            }

            yield _yield_progress(
                "Finetuning complete!",
                100,
                {"completed": True, "result": response},
            )

        except Exception as e:
            yield _yield_progress(
                f"Error: {str(e)}", 100, {"error": True, "detail": str(e)}
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

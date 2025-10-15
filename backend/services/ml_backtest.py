"""
backend/services/ml_backtest.py

Updated ML backtesting service using shared model configuration and results tracking.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi.responses import StreamingResponse
import json
import asyncio

from models.ml_models import (
    ModelConfig, ModelType, TrainingMetrics, TrainingResult, 
    ResultsManager, generate_run_id
)
from models.lstm_model import LSTMStockPredictor
from services.data_provider import get_data_provider
from services.feature_service import get_feature_service, FeatureSet


router = APIRouter()

class BacktestConfig(BaseModel):
    ticker: str
    backtest_mode: str = "standard"
    backtest_days: Optional[int] = 30
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    training_history_days: int = 90
    model_spec: ModelConfig
    retrain_for_each_prediction: bool = False
    notes: Optional[str] = None

class BacktestResult(BaseModel):
    date: str
    actual_close: float
    predicted_close: float
    error: float
    error_pct: float
    training_samples: int

class BacktestResponse(BaseModel):
    status: str
    run_id: str
    ticker: str
    backtest_period: Dict[str, Any]
    predictions: List[BacktestResult]
    summary_metrics: TrainingMetrics
    model_spec: ModelConfig
    timestamp: str

def yield_progress(message: str, progress: float, data: dict = None):
    """Helper to format SSE messages"""
    event_data = {
        "message": message,
        "progress": progress,  # 0-100
        **(data or {})
    }
    return f"data: {json.dumps(event_data)}\n\n"

def create_model(config: ModelConfig):
    """Create a model instance based on configuration"""
    if config.model_type == ModelType.DECISION_TREE:
        return DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state
        )
    elif config.model_type == ModelType.RANDOM_FOREST:
        return RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state
        )
    elif config.model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegression()
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
def train_lstm_for_date(df: pd.DataFrame, end_idx: int, config: BacktestConfig):
    """Train an LSTM model using data up to end_idx"""
    start_idx = max(0, end_idx - config.training_history_days)
    df_train = df.iloc[start_idx:end_idx].copy()
    
    sequence_length = config.model_spec.sequence_length
    min_required = sequence_length + 50
    
    if len(df_train) < min_required:
        return None, 0
    
    predictor = LSTMStockPredictor(
        sequence_length=config.model_spec.sequence_length,
        hidden_size=config.model_spec.hidden_size,
        num_layers=config.model_spec.num_layers,
        dropout=config.model_spec.dropout,
        learning_rate=config.model_spec.learning_rate,
        model_dir='lstm_models_backtest'
    )
    
    backtest_epochs = max(20, config.model_spec.epochs // 3)
    
    try:
        history = predictor.train(
            df_train,
            epochs=backtest_epochs,
            batch_size=config.model_spec.batch_size,
            validation_sequences=config.model_spec.validation_sequences,
            early_stopping_patience=config.model_spec.early_stopping_patience,
            use_validation=config.model_spec.use_validation
        )
        return predictor, len(df_train)
    except Exception as e:
        print(f"LSTM training failed: {str(e)}")
        return None, 0

def make_lstm_prediction(predictor: LSTMStockPredictor, df: pd.DataFrame, predict_idx: int):
    """Make an LSTM prediction for a specific date"""
    try:
        df_pred = df.iloc[:predict_idx].copy()
        predictions = predictor.predict(df_pred, last_sequence_only=True)
        
        if predictions is not None and len(predictions) > 0:
            return predictions[0][0]
        return None
    except Exception as e:
        print(f"LSTM prediction failed: {str(e)}")
        return None

def train_model_for_date(df: pd.DataFrame, end_idx: int, config: BacktestConfig):
    """Train a model using data up to end_idx"""
    start_idx = max(0, end_idx - config.training_history_days)
    df_train = df.iloc[start_idx:end_idx].copy()
    
    feature_service = get_feature_service()
    X, y, feature_cols = feature_service.prepare_for_traditional_ml(
        df_train,
        feature_set=FeatureSet.STANDARD
    )
    
    if len(X) < 20:
        return None, None, None, 0
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = create_model(config.model_spec)
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols, len(X)

def make_prediction(model, scaler, feature_cols, df: pd.DataFrame, predict_idx: int):
    """Make a prediction for a specific date"""
    feature_service = get_feature_service()
    df_pred = df.iloc[:predict_idx].copy()
    
    X_pred, _, _ = feature_service.prepare_for_traditional_ml(
        df_pred,
        feature_set=FeatureSet.STANDARD
    )
    
    X_pred_scaled = scaler.transform(X_pred)
    prediction = model.predict(X_pred_scaled)[0]
    
    return prediction

@router.post("/backtest-model-stream")
async def backtest_model_stream(config: BacktestConfig):
    """Backtest with progress streaming via SSE"""
    
    async def event_generator():
        try:
            yield yield_progress("Starting backtesting...", 0)
            
            # Download data
            await asyncio.sleep(0.1)  # Allow event to be sent
            yield yield_progress("Downloading stock data...", 5)
            
            trading_day_buffer = 1.4
            if config.backtest_mode == "custom":
                start_date = datetime.strptime(config.backtest_start_date, "%Y-%m-%d")
            else:
                calendar_days_needed = int((config.backtest_days + config.training_history_days + 30) * trading_day_buffer)
                start_date = datetime.now() - timedelta(days=calendar_days_needed)
            
            end_date = datetime.now()
            data_provider = get_data_provider()
            df = data_provider.get_stock_history(config.ticker, start=start_date, end=end_date)
            
            min_trading_days = config.training_history_days + 30
            if df.empty or len(df) < min_trading_days:
                yield yield_progress("Error: Insufficient data", 100, {"error": True})
                return
            
            yield yield_progress("Data loaded successfully", 10)
            
            # Determine backtest indices
            if config.backtest_mode == "standard":
                backtest_indices = list(range(len(df) - config.backtest_days, len(df)))
            else:
                backtest_start = datetime.strptime(config.backtest_start_date, "%Y-%m-%d")
                backtest_end = datetime.strptime(config.backtest_end_date, "%Y-%m-%d")
                backtest_indices = [idx for idx, date in enumerate(df.index) 
                                   if backtest_start <= date.to_pydatetime() <= backtest_end]
            
            if len(backtest_indices) < 1:
                yield yield_progress("Error: Insufficient backtest period", 100, {"error": True})
                return
            
            total_predictions = len(backtest_indices)
            yield yield_progress(f"Starting predictions for {total_predictions} days", 15)
            
            # Perform backtesting
            predictions = []
            is_lstm = config.model_spec.model_type == ModelType.LSTM
            
            if is_lstm:
                lstm_predictor = None
                for i, idx in enumerate(backtest_indices):
                    if idx < config.model_spec.sequence_length + 30:
                        continue
                    
                    # Progress: 15% to 85% for predictions
                    progress = 15 + (70 * (i + 1) / total_predictions)
                    
                    if config.retrain_for_each_prediction or lstm_predictor is None:
                        yield yield_progress(
                            f"Training LSTM model for day {i+1}/{total_predictions}...", 
                            progress
                        )
                        lstm_predictor, training_samples = train_lstm_for_date(df, idx, config)
                        if lstm_predictor is None:
                            continue
                    else:
                        training_samples = 0
                    
                    # Make prediction
                    predicted_close = make_lstm_prediction(lstm_predictor, df, idx)
                    actual_close = df['Close'].iloc[idx]
                    error = predicted_close - actual_close
                    error_pct = (error / actual_close) * 100
                    
                    predictions.append(BacktestResult(
                        date=df.index[idx].strftime("%Y-%m-%d"),
                        actual_close=round(actual_close, 2),
                        predicted_close=round(predicted_close, 2),
                        error=round(error, 2),
                        error_pct=round(error_pct, 2),
                        training_samples=training_samples
                    ))
                    
                    if (i + 1) % 5 == 0 or i == total_predictions - 1:
                        yield yield_progress(
                            f"Completed {i+1}/{total_predictions} predictions", 
                            progress
                        )
            else:
                # Traditional ML backtesting
                predictions = []
                model = None
                scaler = None
                feature_cols = None
                for i, idx in enumerate(backtest_indices):
                    progress = 15 + (70 * (i + 1) / total_predictions)
                    
                    if (i + 1) % 10 == 0 or i == 0 or i == total_predictions - 1:
                        yield yield_progress(
                            f"Processing day {i+1}/{total_predictions}...", 
                            progress
                        )
                    if config.retrain_for_each_prediction or model is None:
                        model, scaler, feature_cols, training_samples = train_model_for_date(df, idx, config)
                    if model is None:
                        continue
                    else:
                        training_samples = 0
                    
                    predicted_close = make_prediction(model, scaler, feature_cols, df, idx)

                    if predicted_close is None:
                        continue
                
                    actual_close = df['Close'].iloc[idx]
                    error = predicted_close - actual_close
                    error_pct = (error / actual_close) * 100
                    
                    predictions.append(BacktestResult(
                        date=df.index[idx].strftime("%Y-%m-%d"),
                        actual_close=round(actual_close, 2),
                        predicted_close=round(predicted_close, 2),
                        error=round(error, 2),
                        error_pct=round(error_pct, 2),
                        training_samples=len(df[:idx])
                    ))
            
            yield yield_progress("Calculating metrics...", 90)
            
            # Calculate summary metrics
            actual_values = [p.actual_close for p in predictions]
            predicted_values = [p.predicted_close for p in predictions]
            
            mae = mean_absolute_error(actual_values, predicted_values)
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            r2 = r2_score(actual_values, predicted_values)
            mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
            
            # Directional accuracy
            if len(predictions) > 1:
                correct_direction = 0
                for i in range(1, len(predictions)):
                    actual_direction = predictions[i].actual_close > predictions[i-1].actual_close
                    pred_direction = predictions[i].predicted_close > predictions[i-1].predicted_close
                    if actual_direction == pred_direction:
                        correct_direction += 1
                directional_accuracy = (correct_direction / (len(predictions) - 1)) * 100
            else:
                directional_accuracy = 0.0
            
            summary_metrics = TrainingMetrics(
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                mape=round(mape, 2),
                r2_score=round(r2, 4),
                directional_accuracy=round(directional_accuracy, 2),
                training_samples=predictions[0].training_samples if predictions else 0
            )
            
            # Generate results
            run_id = generate_run_id(config.ticker, "backtest")            
            response = BacktestResponse(
                status="completed",
                run_id=run_id,
                ticker=config.ticker,
                backtest_period={
                    "start": df.index[backtest_indices[0]].strftime("%Y-%m-%d"),
                    "end": df.index[backtest_indices[-1]].strftime("%Y-%m-%d"),
                    "total_days": len(predictions),
                    "training_history_days": config.training_history_days,
                    "retrain_for_each": config.retrain_for_each_prediction
                },
                predictions=predictions,
                summary_metrics=summary_metrics,
                model_spec=config.model_spec,
                timestamp=datetime.now().isoformat()
            )
            
            # Send completion with results
            yield yield_progress(
                "Backtesting complete!", 
                100, 
                {"completed": True, "result": response.dict()}
            )
            
        except Exception as e:
            yield yield_progress(
                f"Error: {str(e)}", 
                100, 
                {"error": True, "detail": str(e)}
            )
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import sklearn
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Optional
import pandas as pd

from models.ml_models import (
    ModelConfig, ModelType, TrainingMetrics, PredictionResult,
    TrainingResult, ResultsManager, generate_run_id
)

from models.lstm_model import LSTMStockPredictor
from services.data_provider import get_data_provider
from services.feature_service import get_feature_service, FeatureSet
from services.metrics_service import get_metrics_service
import torch

router = APIRouter()

class TrainingConfig(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    model_spec: ModelConfig
    notes: Optional[str] = None

class TrainingResponse(BaseModel):
    status: str
    run_id: str
    ticker: str
    training_metrics: TrainingMetrics
    prediction: PredictionResult
    feature_importance: Dict[str, float]
    training_period: Dict[str, Any]
    model_spec: ModelConfig
    timestamp: str

def train_lstm_model(df: pd.DataFrame, config: TrainingConfig):
    """
    Updated LSTM training with directional loss support
    """
    
    # Create LSTM predictor with ALL the new parameters
    predictor = LSTMStockPredictor(
        sequence_length=config.model_spec.sequence_length,
        hidden_size=config.model_spec.hidden_size,
        num_layers=config.model_spec.num_layers,
        dropout=config.model_spec.dropout,
        learning_rate=config.model_spec.learning_rate,
        model_dir='lstm_models',
        
        # NEW: Enhanced LSTM parameters (add these if they exist in ModelConfig)
        bidirectional=getattr(config.model_spec, 'bidirectional', True),
        use_layer_norm=getattr(config.model_spec, 'use_layer_norm', True),
        use_residual=getattr(config.model_spec, 'use_residual', True),
        weight_decay=getattr(config.model_spec, 'weight_decay', 1e-5),
        gradient_clip_norm=getattr(config.model_spec, 'gradient_clip_norm', 1.0),
        
        # NEW: Directional loss configuration
        use_directional_loss=getattr(config.model_spec, 'use_directional_loss', False),
        directional_loss_config={
            'loss_type': getattr(config.model_spec, 'directional_loss_type', 'standard'),
            'price_weight': getattr(config.model_spec, 'price_weight', 0.7),
            'direction_weight': getattr(config.model_spec, 'direction_weight', 0.3),
            'direction_threshold': getattr(config.model_spec, 'direction_threshold', 0.0),
            # Optional focal loss parameters
            'focal_alpha': getattr(config.model_spec, 'focal_alpha', 0.25),
            'focal_gamma': getattr(config.model_spec, 'focal_gamma', 2.0),
            # Optional adaptive loss parameters  
            'initial_price_weight': getattr(config.model_spec, 'initial_price_weight', 0.7),
            'target_direction_accuracy': getattr(config.model_spec, 'target_direction_accuracy', 60.0),
            'adaptation_rate': getattr(config.model_spec, 'adaptation_rate', 0.01)
        }
    )
    
    # Train the model with enhanced parameters
    history = predictor.train(
        df, 
        epochs=config.model_spec.epochs,
        batch_size=config.model_spec.batch_size,
        validation_sequences=config.model_spec.validation_sequences,
        early_stopping_patience=config.model_spec.early_stopping_patience,
        use_validation=config.model_spec.use_validation,
        
        # NEW: Add these if they exist in ModelConfig
        lr_patience=getattr(config.model_spec, 'lr_patience', 7),
        lr_factor=getattr(config.model_spec, 'lr_factor', 0.5),
        min_lr=getattr(config.model_spec, 'min_lr', 1e-6)
    )
    
    # Make prediction for next day
    next_day_pred = predictor.predict(df, last_sequence_only=True)
    next_day_prediction = next_day_pred[0][0]
    
    # Get metrics from training history or validation
    if config.model_spec.use_validation:
        validation_metrics = predictor.get_validation_metrics()
        rmse = validation_metrics['rmse']
        mae = validation_metrics['mae']
        r2 = validation_metrics['r2']
        mape = validation_metrics['mape']
    else:
        # Fallback to final training loss
        final_loss = history['train_loss'][-1]
        rmse = np.sqrt(final_loss)
        mae = final_loss
        r2 = 0.0
        mape = 0.0
    
    # Save model
    model_path = predictor.save_model(
        config.ticker, 
        metadata={
            'training_samples': len(df),
            'use_directional_loss': predictor.use_directional_loss,
            'directional_loss_config': predictor.directional_loss_config
        }
    )
    
    # Create training period info
    training_period = {
        "start": config.start_date,
        "end": config.end_date,
        "days": len(df)
    }
    
    # Save training result
    training_metrics = TrainingMetrics(
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2_score=round(r2, 4),
        mape=round(mape, 4),
        training_samples=len(df),
        # NEW: Add directional accuracy if available
        directional_accuracy=round(history.get('train_direction_accuracy', [0])[-1], 2) if history.get('train_direction_accuracy') else None
    )
    
    # Get last close price for prediction
    last_close = df['Close'].iloc[-1]
    predicted_change = next_day_prediction - last_close
    predicted_change_pct = (predicted_change / last_close) * 100
    
    # Generate run ID and prediction date
    run_id = generate_run_id(config.ticker, "training")
    last_date = df.index[-1]
    prediction_date = last_date + timedelta(days=1)
    while prediction_date.weekday() >= 5:
        prediction_date += timedelta(days=1)
    
    prediction = PredictionResult(
        date=prediction_date.strftime("%Y-%m-%d"),
        predicted_close=round(next_day_prediction, 2),
        last_close=round(last_close, 2),
        predicted_change=round(predicted_change, 2),
        predicted_change_pct=round(predicted_change_pct, 2)
    )
    
    # Save result
    training_result = TrainingResult(
        run_id=run_id,
        run_type="training",
        timestamp=datetime.now().isoformat(),
        ticker=config.ticker,
        model_spec=config.model_spec,
        training_period=training_period,
        metrics=training_metrics,
        prediction=prediction,
        feature_importance={},  # LSTM doesn't have traditional feature importance
        notes=config.notes
    )
    
    results_manager = ResultsManager()
    results_manager.save_result(training_result)

    return predictor, rmse, mae, r2, mape, next_day_prediction, model_path

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
    elif config.model_type == ModelType.LSTM:
        return LSTMStockPredictor(
            sequence_length=config.sequence_length,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            model_dir='lstm_models'
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

@router.post("/train-model", response_model=TrainingResponse)
async def train_model(config: TrainingConfig):
    """Train a machine learning model for stock price prediction"""
    try:
        # Validate dates
        start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if (end_date - start_date).days < 30:
            raise HTTPException(status_code=400, detail="Training period must be at least 30 days")
        
        data_provider = get_data_provider()
        df = data_provider.get_stock_history(
            config.ticker,
            start=start_date,
            end=end_date + timedelta(days=5)
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {config.ticker}")
        
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data for training")
        
        # Branch based on model type
        if config.model_spec.model_type == ModelType.LSTM:
            # Train LSTM model
            predictor, rmse, mae, r2, mape, next_day_prediction, model_path = train_lstm_model(df, config)
            
            # Get last close price
            last_close = df['Close'].iloc[-1]
            predicted_change = next_day_prediction - last_close
            predicted_change_pct = (predicted_change / last_close) * 100
            
            # Generate run ID
            run_id = generate_run_id(config.ticker, "training")
            
            # Get prediction date
            last_date = df.index[-1]
            prediction_date = last_date + timedelta(days=1)
            while prediction_date.weekday() >= 5:
                prediction_date += timedelta(days=1)
            
            # Create response
            training_metrics = TrainingMetrics(
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                r2_score=round(r2, 4),
                mape=round(mape, 4),
                training_samples=len(df)
            )
            
            prediction = PredictionResult(
                date=prediction_date.strftime("%Y-%m-%d"),
                predicted_close=round(next_day_prediction, 2),
                last_close=round(last_close, 2),
                predicted_change=round(predicted_change, 2),
                predicted_change_pct=round(predicted_change_pct, 2)
            )
            
            return TrainingResponse(
                status="success",
                run_id=run_id,
                ticker=config.ticker,
                training_metrics=training_metrics,
                prediction=prediction,
                feature_importance={},  # LSTM doesn't have traditional feature importance
                training_period={
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d"),
                    "days": len(df)
                },
                model_spec=config.model_spec,
                timestamp=datetime.now().isoformat()
            )
            
        else:
          # Traditional ML model training

            feature_service = get_feature_service()
            
            # Prepare data for traditional ML
            X, y, feature_cols = feature_service.prepare_for_traditional_ml(
                df, 
                feature_set=FeatureSet.STANDARD
            )
          
            if len(X) < 20:
                raise HTTPException(status_code=400, detail="Insufficient data after feature engineering")
            
            # Split data (use last row for prediction, rest for training)
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_predict = X.iloc[[-1]]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_predict_scaled = scaler.transform(X_predict)
            
            # Train model
            model = create_model(config.model_spec)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions on training set for metrics
            y_train_pred = model.predict(X_train_scaled)
            
            metrics_service = get_metrics_service()
            training_metrics = metrics_service.calculate_regression_metrics(
                actual=y_train,
                predicted=y_train_pred,
                training_samples=len(X_train)
            )
            
            # Make prediction for next day
            next_day_prediction = model.predict(X_predict_scaled)[0]
            
            # Get last actual close price
            last_close = df['Close'].iloc[-1]
            predicted_change = next_day_prediction - last_close
            predicted_change_pct = (predicted_change / last_close) * 100
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
                feature_importance = dict(sorted(feature_importance.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10])
            else:
                feature_importance = {}
            
            # Generate run ID
            run_id = generate_run_id(config.ticker, "training")
            
            # Get prediction date
            last_date = df.index[-1]
            prediction_date = last_date + timedelta(days=1)
            while prediction_date.weekday() >= 5:
                prediction_date += timedelta(days=1)
            
            
            prediction = PredictionResult(
                date=prediction_date.strftime("%Y-%m-%d"),
                predicted_close=round(next_day_prediction, 2),
                last_close=round(last_close, 2),
                predicted_change=round(predicted_change, 2),
                predicted_change_pct=round(predicted_change_pct, 2)
            )
            
            training_period = {
                "start": config.start_date,
                "end": config.end_date,
                "days": len(df)
            }
            
            # Save result to file
            training_result = TrainingResult(
                run_id=run_id,
                run_type="training",
                timestamp=datetime.now().isoformat(),
                ticker=config.ticker,
                model_spec=config.model_spec,
                training_period=training_period,
                metrics=training_metrics,
                prediction=prediction,
                feature_importance={k: round(v, 4) for k, v in feature_importance.items()},
                notes=config.notes
            )
            
            results_manager = ResultsManager()
            results_manager.save_result(training_result)
            
            return TrainingResponse(
                status="success",
                run_id=run_id,
                ticker=config.ticker,
                training_metrics=training_metrics,
                prediction=prediction,
                feature_importance={k: round(v, 4) for k, v in feature_importance.items()},
                training_period=training_period,
                model_spec=config.model_spec,
                timestamp=datetime.now().isoformat()
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.get("/available-tickers")
async def get_available_tickers():
    """Get list of available stock tickers for training"""
    try:
        from data.tickers import STOCK_TICKERS
        return {"tickers": list(STOCK_TICKERS)}
    except Exception as e:
        # Fallback to some default tickers if the import fails
        return {"tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]}
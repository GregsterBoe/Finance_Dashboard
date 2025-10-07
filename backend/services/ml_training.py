from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import sklearn
import yfinance as yf
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

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators and features for training"""
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    # Volatility
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    
    # Price momentum
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    # Volume features
    df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_5']
    
    # High-Low range
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    
    # Lag features
    df['close_lag_1'] = df['Close'].shift(1)
    df['close_lag_2'] = df['Close'].shift(2)
    df['close_lag_3'] = df['Close'].shift(3)
    
    # Target variable (next day's closing price)
    df['target'] = df['Close'].shift(-1)
    
    return df

def prepare_training_data(df: pd.DataFrame):
    """Prepare features and target for training"""
    # Drop rows with NaN values
    df_clean = df.dropna()
    
    # Define feature columns (exclude target and original OHLCV)
    feature_cols = [col for col in df_clean.columns 
                   if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    return X, y, feature_cols

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
        
        # Fetch stock data
        stock = yf.Ticker(config.ticker)
        df = stock.history(start=start_date, end=end_date + timedelta(days=5))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {config.ticker}")
        
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data for training")
        
        # Create features
        df_features = create_features(df)
        
        # Prepare training data
        X, y, feature_cols = prepare_training_data(df_features)
        
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
        
        # Calculate training metrics
        mse = mean_squared_error(y_train, y_train_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_train, y_train_pred)
        r2 = r2_score(y_train, y_train_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        
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
        
        # Create response objects
        training_metrics = TrainingMetrics(
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r2_score=round(r2, 4),
            mape=round(mape, 4),
            training_samples=len(X_train)
        )
        
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
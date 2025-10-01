from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Optional
import pandas as pd

router = APIRouter()

class BacktestConfig(BaseModel):
    ticker: str
    backtest_mode: str = "standard"  # "standard" or "custom"
    backtest_days: Optional[int] = 30  # For standard mode
    backtest_start_date: Optional[str] = None  # For custom mode
    backtest_end_date: Optional[str] = None  # For custom mode
    training_history_days: int = 90
    model_type: str = "decision_tree"
    max_depth: Optional[int] = 5
    min_samples_split: Optional[int] = 2
    min_samples_leaf: Optional[int] = 1
    retrain_for_each_prediction: bool = False

class BacktestResult(BaseModel):
    date: str
    actual_close: float
    predicted_close: float
    error: float
    error_pct: float
    training_samples: int

class BacktestResponse(BaseModel):
    status: str
    ticker: str
    backtest_period: Dict[str, Any]
    predictions: List[BacktestResult]
    summary_metrics: Dict[str, float]
    model_parameters: Dict[str, Any] 
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
    df_clean = df.dropna()
    
    feature_cols = [col for col in df_clean.columns 
                   if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    return X, y, feature_cols

def train_model_for_date(df: pd.DataFrame, end_idx: int, config: BacktestConfig):
    """Train a model using data up to end_idx"""
    # Get training data (use last training_history_days)
    start_idx = max(0, end_idx - config.training_history_days)
    df_train = df.iloc[start_idx:end_idx].copy()
    
    # Create features
    df_features = create_features(df_train)
    
    # Prepare training data
    X, y, feature_cols = prepare_training_data(df_features)
    
    if len(X) < 20:
        return None, None, None, 0
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if config.model_type == "decision_tree":
        model = DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    model.fit(X_scaled, y)
    
    return model, scaler, feature_cols, len(X)

def make_prediction(model, scaler, feature_cols, df: pd.DataFrame, predict_idx: int):
    """Make a prediction for a specific date"""
    # Get data up to (but not including) the prediction date
    df_pred = df.iloc[:predict_idx].copy()
    
    # Create features
    df_features = create_features(df_pred)
    df_clean = df_features.dropna()
    
    if len(df_clean) == 0:
        return None
    
    # Get the last row for prediction
    X_pred = df_clean[feature_cols].iloc[[-1]]
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make prediction
    prediction = model.predict(X_pred_scaled)[0]
    
    return prediction

@router.post("/backtest-model", response_model=BacktestResponse)
async def backtest_model(config: BacktestConfig):
    """Backtest a machine learning model for stock price prediction"""
    try:
        # Fetch historical data (need extra buffer for feature creation)
        stock = yf.Ticker(config.ticker)
        
        # Determine backtest period
        if config.backtest_mode == "standard":
            if not config.backtest_days or config.backtest_days < 1:
                raise HTTPException(status_code=400, detail="Backtest days must be at least 1")
            
            end_date = datetime.now()
            # Add buffer for training history and feature creation
            total_days_needed = config.backtest_days + config.training_history_days + 60
            start_date = end_date - timedelta(days=total_days_needed)
            
        elif config.backtest_mode == "custom":
            if not config.backtest_start_date or not config.backtest_end_date:
                raise HTTPException(status_code=400, detail="Custom mode requires start and end dates")
            
            backtest_start = datetime.strptime(config.backtest_start_date, "%Y-%m-%d")
            backtest_end = datetime.strptime(config.backtest_end_date, "%Y-%m-%d")
            
            if backtest_start >= backtest_end:
                raise HTTPException(status_code=400, detail="Start date must be before end date")
            
            # Add buffer for training history and feature creation
            start_date = backtest_start - timedelta(days=config.training_history_days + 60)
            end_date = backtest_end + timedelta(days=5)
        else:
            raise HTTPException(status_code=400, detail="Invalid backtest mode")
        
        # Fetch data
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {config.ticker}")
        
        # Determine backtest indices
        if config.backtest_mode == "standard":
            # Get last N trading days
            backtest_indices = list(range(len(df) - config.backtest_days, len(df)))
        else:
            # Filter to custom date range
            backtest_start = datetime.strptime(config.backtest_start_date, "%Y-%m-%d")
            backtest_end = datetime.strptime(config.backtest_end_date, "%Y-%m-%d")
            backtest_indices = []
            
            for idx, date in enumerate(df.index):
                if backtest_start <= date.to_pydatetime() <= backtest_end:
                    backtest_indices.append(idx)
        
        if len(backtest_indices) < 1:
            raise HTTPException(status_code=400, detail="Insufficient data for backtesting period")
        
        # Perform backtesting
        predictions = []
        model = None
        scaler = None
        feature_cols = None
        
        for idx in backtest_indices:
            # Skip if not enough training data
            if idx < 30:
                continue
            
            # Train or reuse model
            if config.retrain_for_each_prediction or model is None:
                model, scaler, feature_cols, training_samples = train_model_for_date(df, idx, config)
                if model is None:
                    continue
            else:
                training_samples = 0  # Not retraining
            
            # Make prediction
            predicted_close = make_prediction(model, scaler, feature_cols, df, idx)
            
            if predicted_close is None:
                continue
            
            # Get actual value
            actual_close = df['Close'].iloc[idx]
            
            # Calculate error
            error = predicted_close - actual_close
            error_pct = (error / actual_close) * 100
            
            predictions.append(BacktestResult(
                date=df.index[idx].strftime("%Y-%m-%d"),
                actual_close=round(float(actual_close), 2),
                predicted_close=round(float(predicted_close), 2),
                error=round(float(error), 2),
                error_pct=round(float(error_pct), 2),
                training_samples=training_samples
            ))
        
        if len(predictions) == 0:
            raise HTTPException(status_code=400, detail="No valid predictions could be made")
        
        # Calculate summary metrics
        errors = [p.error for p in predictions]
        error_pcts = [p.error_pct for p in predictions]
        actuals = [p.actual_close for p in predictions]
        predicted = [p.predicted_close for p in predictions]
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        mape = np.mean(np.abs(error_pcts))
        r2 = r2_score(actuals, predicted)
        
        # Directional accuracy (did we predict the right direction?)
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
        
        summary_metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "r2_score": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 2),
            "total_predictions": len(predictions),
            "avg_error": round(np.mean(errors), 4),
            "avg_error_pct": round(np.mean(error_pcts), 4)
        }
        
        # Determine actual backtest period from predictions
        backtest_period = {
            "start": predictions[0].date,
            "end": predictions[-1].date,
            "total_days": len(predictions),
            "training_history_days": config.training_history_days,
            "retrain_for_each": config.retrain_for_each_prediction
        }
        
        return BacktestResponse(
            status="success",
            ticker=config.ticker,
            backtest_period=backtest_period,
            predictions=predictions,
            summary_metrics=summary_metrics,
            model_parameters={
                "model_type": config.model_type,
                "max_depth": config.max_depth,
                "min_samples_split": config.min_samples_split,
                "min_samples_leaf": config.min_samples_leaf
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting error: {str(e)}")
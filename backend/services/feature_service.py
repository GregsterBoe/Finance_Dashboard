"""
backend/services/feature_service.py

Unified Feature Management Service for ML Stock Prediction

This service centralizes ALL feature engineering for both LSTM and traditional ML models.
It provides a consistent interface for creating technical indicators and preparing data
for different model types.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from enum import Enum


class FeatureSet(Enum):
    """Types of feature sets available"""
    BASIC = "basic"           # Basic price and volume features
    STANDARD = "standard"     # Standard technical indicators
    ADVANCED = "advanced"     # Advanced features including RSI, MACD, etc.


class FeatureService:
    """
    Unified service for creating and managing features for stock prediction models.
    
    This service handles:
    1. Technical indicator calculation
    2. Feature engineering for both LSTM and traditional ML
    3. Target variable creation
    4. Feature column identification
    """
    
    def __init__(self):
        """Initialize the feature service"""
        self.raw_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    def create_features(
        self, 
        df: pd.DataFrame, 
        feature_set: FeatureSet = FeatureSet.STANDARD,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Create technical indicators and features from raw OHLCV data.
        
        This is the main feature creation method used by ALL models (LSTM and traditional ML).
        
        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
            feature_set: Which set of features to create (BASIC, STANDARD, ADVANCED)
            include_target: Whether to include the target variable (next day's close)
        
        Returns:
            DataFrame with original data + engineered features
        """
        df = df.copy()
        
        # BASIC FEATURES - Always included
        df = self._create_basic_features(df)
        
        # STANDARD FEATURES - Technical indicators
        if feature_set in [FeatureSet.STANDARD, FeatureSet.ADVANCED]:
            df = self._create_standard_features(df)
        
        # ADVANCED FEATURES - Additional indicators
        if feature_set == FeatureSet.ADVANCED:
            df = self._create_advanced_features(df)
        
        # Target variable (next day's closing price)
        if include_target:
            df['target'] = df['Close'].shift(-1)
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic price and volume features"""
        # Returns and log returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-8)
        df['close_open_ratio'] = df['Close'] / (df['Open'] + 1e-8)
        
        # Lagged close prices
        df['close_lag_1'] = df['Close'].shift(1)
        df['close_lag_2'] = df['Close'].shift(2)
        df['close_lag_3'] = df['Close'].shift(3)
        
        return df
    
    def _create_standard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create standard technical indicators with NaN handling"""
        
        # Simple Moving Averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        
        # Fill initial NaN values with the close price for SMAs
        df['sma_5'] = df['sma_5'].fillna(df['Close'])
        df['sma_10'] = df['sma_10'].fillna(df['Close'])
        df['sma_20'] = df['sma_20'].fillna(df['Close'])
        
        # Exponential Moving Averages
        df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # EMA handles NaN better, but still fill just in case
        df['ema_5'] = df['ema_5'].fillna(df['Close'])
        df['ema_10'] = df['ema_10'].fillna(df['Close'])
        
        # Volatility (with minimum window handling)
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        
        # Fill volatility NaN with small default value
        df['volatility_5'] = df['volatility_5'].fillna(0.01)
        df['volatility_10'] = df['volatility_10'].fillna(0.01)
        
        # Price momentum
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Fill momentum NaN with 0
        df['momentum_5'] = df['momentum_5'].fillna(0)
        df['momentum_10'] = df['momentum_10'].fillna(0)
        
        # Volume features
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma_5'] + 1e-8)
        
        # Fill volume features
        df['volume_sma_5'] = df['volume_sma_5'].fillna(df['Volume'])
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        return df

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators with proper NaN handling"""
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill RSI NaN with neutral value (50)
        df['rsi'] = df['rsi'].fillna(50)
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Fill MACD NaN with 0
        df['macd'] = df['macd'].fillna(0)
        df['macd_signal'] = df['macd_signal'].fillna(0)
        df['macd_diff'] = df['macd_diff'].fillna(0)
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Fill Bollinger Bands NaN
        df['bb_middle'] = df['bb_middle'].fillna(df['Close'])
        df['bb_upper'] = df['bb_upper'].fillna(df['Close'])
        df['bb_lower'] = df['bb_lower'].fillna(df['Close'])
        df['bb_width'] = df['bb_width'].fillna(0.01)
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Fill ATR NaN with median value
        df['atr'] = df['atr'].fillna(df['atr'].median())
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding raw OHLCV and target).
        
        Args:
            df: DataFrame with features
        
        Returns:
            List of feature column names
        """
        exclude_cols = self.raw_columns + ['target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_for_traditional_ml(
        self, 
        df: pd.DataFrame,
        feature_set: FeatureSet = FeatureSet.STANDARD
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for traditional ML models (Random Forest, Decision Tree, etc.).
        
        This method:
        1. Creates features
        2. Removes NaN values
        3. Separates features and target
        4. Returns clean data ready for sklearn models
        
        Args:
            df: Raw OHLCV DataFrame
            feature_set: Which feature set to use
        
        Returns:
            X: Feature DataFrame
            y: Target Series
            feature_cols: List of feature column names
        """
        # Create features
        df_features = self.create_features(df, feature_set=feature_set, include_target=True)
        
        # Remove NaN values
        df_clean = df_features.dropna()
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df_clean)
        
        # Separate features and target
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        return X, y, feature_cols
    
    def prepare_for_lstm(
        self,
        df: pd.DataFrame,
        feature_set: FeatureSet = FeatureSet.ADVANCED
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for LSTM models.
        
        This method:
        1. Creates features (typically with ADVANCED set for LSTM)
        2. Returns feature DataFrame with target
        3. Returns feature column names
        
        Note: Sequence creation and scaling is handled by the LSTM model itself.
        
        Args:
            df: Raw OHLCV DataFrame
            feature_set: Which feature set to use (default ADVANCED for LSTM)
        
        Returns:
            df_features: DataFrame with features and target
            feature_cols: List of feature column names
        """
        # Create features (LSTM typically uses more advanced features)
        df_features = self.create_features(df, feature_set=feature_set, include_target=True)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df_features)
        
        return df_features, feature_cols
    
    def get_feature_info(self, feature_set: FeatureSet = FeatureSet.STANDARD) -> dict:
        """
        Get information about what features are included in each feature set.
        
        Args:
            feature_set: Which feature set to describe
        
        Returns:
            Dictionary with feature set information
        """
        feature_info = {
            FeatureSet.BASIC: {
                "description": "Basic price and volume features",
                "features": [
                    "returns", "log_returns",
                    "high_low_ratio", "close_open_ratio",
                    "close_lag_1", "close_lag_2", "close_lag_3"
                ],
                "count": 7
            },
            FeatureSet.STANDARD: {
                "description": "Standard technical indicators + basic features",
                "features": [
                    # Basic features
                    "returns", "log_returns", "high_low_ratio", "close_open_ratio",
                    "close_lag_1", "close_lag_2", "close_lag_3",
                    # Standard indicators
                    "sma_5", "sma_10", "sma_20",
                    "ema_5", "ema_10",
                    "volatility_5", "volatility_10",
                    "momentum_5", "momentum_10",
                    "volume_sma_5", "volume_ratio"
                ],
                "count": 18
            },
            FeatureSet.ADVANCED: {
                "description": "Advanced technical indicators + standard + basic features",
                "features": [
                    # All standard features plus:
                    "rsi", "macd", "macd_signal", "macd_diff",
                    "bb_middle", "bb_upper", "bb_lower", "bb_width",
                    "atr"
                ],
                "additional_count": 9,
                "total_count": 27
            }
        }
        
        return feature_info.get(feature_set, {})


# Singleton instance for global use
_feature_service_instance = None


def get_feature_service() -> FeatureService:
    """
    Get the global feature service instance.
    
    Returns:
        FeatureService singleton instance
    """
    global _feature_service_instance
    if _feature_service_instance is None:
        _feature_service_instance = FeatureService()
    return _feature_service_instance
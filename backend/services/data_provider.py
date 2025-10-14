"""
Unified Data Provider Service

This service centralizes all data access for ML algorithms and other services.
Currently supports stock data from yfinance, designed to be extended with
additional data sources like social media sentiment and macro economic data.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from enum import Enum


class DataSource(Enum):
    """Available data sources"""
    STOCK_DATA = "stock"
    SENTIMENT = "sentiment"  # Future: social media sentiment
    MACRO = "macro"  # Future: macro economic indicators
    

class StockDataProvider:
    """Provider for stock market data from yfinance"""
    
    def __init__(self):
        self._cache = {}
    
    def get_history(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical stock data
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start: Start date for historical data
            end: End date for historical data
            period: Period string (e.g., '1mo', '1y') - alternative to start/end
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        """
        stock = yf.Ticker(ticker)
        
        if period:
            return stock.history(period=period, interval=interval)
        else:
            return stock.history(start=start, end=end, interval=interval)
    
    def get_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information and fundamentals
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary containing stock info (market cap, PE ratio, etc.)
        """
        stock = yf.Ticker(ticker)
        return stock.info
    
    def get_ticker(self, ticker: str) -> yf.Ticker:
        """
        Get yfinance Ticker object for advanced usage
        
        Args:
            ticker: Stock symbol
            
        Returns:
            yfinance Ticker object
        """
        return yf.Ticker(ticker)
    
    def download_multiple(
        self,
        tickers: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download data for multiple tickers efficiently
        
        Args:
            tickers: List of stock symbols
            start: Start date
            end: End date
            period: Period string (alternative to start/end)
            
        Returns:
            DataFrame with multi-index columns (ticker, data_type)
        """
        if period:
            return yf.download(tickers, period=period, group_by='ticker')
        else:
            return yf.download(tickers, start=start, end=end, group_by='ticker')


class SentimentDataProvider:
    """Provider for social media sentiment data (placeholder for future implementation)"""
    
    def get_sentiment(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get sentiment data for a ticker
        
        Future implementation will include:
        - Twitter/X sentiment analysis
        - Reddit WallStreetBets sentiment
        - News sentiment from various sources
        """
        raise NotImplementedError("Sentiment data provider not yet implemented")


class MacroDataProvider:
    """Provider for macro economic data (placeholder for future implementation)"""
    
    def get_indicator(self, indicator: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get macro economic indicator data
        
        Future implementation will include:
        - Federal funds rate
        - Inflation data (CPI, PPI)
        - GDP growth
        - Unemployment rate
        - Treasury yields
        """
        raise NotImplementedError("Macro data provider not yet implemented")


class UnifiedDataProvider:
    """
    Unified interface for all data sources
    
    This is the main class that ML algorithms and other services should use.
    It provides a single point of access to all data sources.
    """
    
    def __init__(self):
        self.stock = StockDataProvider()
        self.sentiment = SentimentDataProvider()
        self.macro = MacroDataProvider()
    
    # Convenience methods for stock data (most common use case)
    def get_stock_history(
        self,
        ticker: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical stock data - convenience wrapper"""
        return self.stock.get_history(ticker, start, end, period, interval)
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information - convenience wrapper"""
        return self.stock.get_info(ticker)
    
    def get_stock_ticker(self, ticker: str) -> yf.Ticker:
        """Get yfinance ticker object - convenience wrapper"""
        return self.stock.get_ticker(ticker)
    
    # Future: Combined data methods
    def get_ml_dataset(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        include_sentiment: bool = False,
        include_macro: bool = False
    ) -> pd.DataFrame:
        """
        Get a complete dataset for ML training
        
        Future implementation will merge:
        - Stock price data
        - Technical indicators
        - Sentiment data (if requested)
        - Macro economic data (if requested)
        
        Returns a single DataFrame with all features aligned by date
        """
        # For now, just return stock data
        df = self.stock.get_history(ticker, start=start_date, end=end_date)
        
        # Future: Add sentiment and macro data when available
        # if include_sentiment:
        #     sentiment_df = self.sentiment.get_sentiment(ticker, start_date, end_date)
        #     df = df.join(sentiment_df, how='left')
        #
        # if include_macro:
        #     macro_df = self.macro.get_indicator('fed_rate', start_date, end_date)
        #     df = df.join(macro_df, how='left')
        
        return df


# Singleton instance for global use
_data_provider_instance = None

def get_data_provider() -> UnifiedDataProvider:
    """
    Get the global data provider instance
    
    Returns:
        UnifiedDataProvider singleton instance
    """
    global _data_provider_instance
    if _data_provider_instance is None:
        _data_provider_instance = UnifiedDataProvider()
    return _data_provider_instance
from fastapi import APIRouter, HTTPException
import yfinance as yf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime

router = APIRouter()

MAJOR_MARKETS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC", 
    "DOW JONES": "^DJI",
    "RUSSELL 2000": "^RUT",
    "DAX": "^GDAXI",
    "NIKKEI": "^N225",
    "FTSE 100": "^FTSE",
    "SHANGHAI COMP": "000001.SS",
    "HANG SENG": "^HSI"
}

# Thread pool for running synchronous yfinance calls
executor = ThreadPoolExecutor(max_workers=10)

async def get_market_data(ticker: str, name: str):
    """Fetch market data asynchronously"""
    try:
        # Run synchronous yfinance call in thread pool
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            executor, 
            lambda: yf.download(ticker, period="5d", interval="1d", progress=False)
        )
        
        if df.empty or len(df) < 2:
            print(f"No data or insufficient data for {name} ({ticker})")
            return None
            
        # Get the latest data - use .iloc to avoid Series issues
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
        
        # Extract values properly using .iloc[0] or direct access
        current_close = float(latest["Close"])
        previous_close = float(prev_close)
        
        price_change = current_close - previous_close
        percent_change = (price_change / previous_close) * 100 if previous_close != 0 else 0
        
        # Fix the volume extraction
        volume = int(latest["Volume"].iloc[0]) if hasattr(latest["Volume"], 'iloc') else int(latest["Volume"])
        
        return {
            "name": name,
            "ticker": ticker,
            "price": round(current_close, 2),
            "change": round(price_change, 2),
            "percent_change": round(percent_change, 2),
            "volume": volume,
            "last_updated": latest.name.strftime("%Y-%m-%d %H:%M:%S") if hasattr(latest.name, 'strftime') else str(latest.name),
            "previous_close": round(previous_close, 2),
            "currency": "USD"  # Default currency
        }
        
    except Exception as e:
        print(f"Error fetching {name} ({ticker}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@router.get("/market-overview")
async def market_overview():
    """Get overview of major market indices"""
    try:
        # Fetch all market data concurrently
        tasks = [
            get_market_data(ticker, name) 
            for name, ticker in MAJOR_MARKETS.items()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and create response
        market_data = [result for result in results if result is not None]
        
        # Calculate overall market status
        up_markets = sum(1 for market in market_data if market and market.get("change", 0) > 0)
        down_markets = sum(1 for market in market_data if market and market.get("change", 0) < 0)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "markets": market_data,
            "summary": {
                "total_markets": len(market_data),
                "up_markets": up_markets,
                "down_markets": down_markets,
                "neutral_markets": len(market_data) - up_markets - down_markets
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching market data: {str(e)}"
        )

from fastapi import APIRouter, HTTPException
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any

router = APIRouter()

MAJOR_MARKETS = {
    # US Markets
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC", 
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    
    # European Markets
    "FTSE 100 (UK)": "^FTSE",
    "DAX (Germany)": "^GDAXI", 
    "CAC 40 (France)": "^FCHI",
    "IBEX 35 (Spain)": "^IBEX",
    "FTSE MIB (Italy)": "FTSEMIB.MI",
    "AEX (Netherlands)": "^AEX",
    "SMI (Switzerland)": "^SSMI",
    "OMXS30 (Sweden)": "^OMX",
    "Euro Stoxx 50": "^SX5E",
    
    # Asian Markets
    "Nikkei 225 (Japan)": "^N225",
    "Hang Seng (Hong Kong)": "^HSI",
    "Shanghai Composite": "000001.SS",
    "KOSPI (South Korea)": "^KS11",
    "SENSEX (India)": "^BSESN",
    "Nifty 50 (India)": "^NSEI",
    "ASX 200 (Australia)": "^AXJO",
    
    # Other Major Markets
    "TSX (Canada)": "^GSPTSE",
    "BOVESPA (Brazil)": "^BVSP",
    "MOEX (Russia)": "IMOEX.ME",
    "JSE (South Africa)": "^JN0U.JO",
}

def get_market_region(ticker: str) -> str:
    """Determine market region based on ticker symbol"""
    if ticker in ["^GSPC", "^IXIC", "^DJI", "^RUT", "SPY", "QQQ"]:
        return "North America"
    elif ticker in ["^FTSE", "^GDAXI", "^FCHI", "^IBEX", "FTSEMIB.MI", "^AEX", "^SSMI", "^OMX", "^SX5E"]:
        return "Europe"
    elif ticker in ["^N225", "^HSI", "000001.SS", "^KS11", "^BSESN", "^NSEI", "^AXJO"]:
        return "Asia-Pacific"
    else:
        return "Other"

def get_market_currency(ticker: str) -> str:
    """Get appropriate currency symbol based on market"""
    currency_map = {
        "^GSPC": "USD", "^IXIC": "USD", "^DJI": "USD", "^RUT": "USD", "SPY": "USD", "QQQ": "USD",
        "^FTSE": "GBP", "^GDAXI": "EUR", "^FCHI": "EUR", "^IBEX": "EUR", "FTSEMIB.MI": "EUR",
        "^AEX": "EUR", "^SSMI": "CHF", "^OMX": "SEK", "^SX5E": "EUR",
        "^N225": "JPY", "^HSI": "HKD", "000001.SS": "CNY", "^KS11": "KRW",
        "^BSESN": "INR", "^NSEI": "INR", "^AXJO": "AUD",
        "^GSPTSE": "CAD", "^BVSP": "BRL", "IMOEX.ME": "RUB", "^JN0U.JO": "ZAR"
    }
    return currency_map.get(ticker, "USD")

def calculate_market_summary(market_data):
    """Calculate market summary statistics based on actual market changes"""
    up_markets = sum(1 for m in market_data if m["change"] > 0)
    down_markets = sum(1 for m in market_data if m["change"] < 0)
    neutral_markets = sum(1 for m in market_data if m["change"] == 0)
    
    # Regional breakdown
    regions = {}
    for market in market_data:
        region = market.get("region", "Other")
        if region not in regions:
            regions[region] = {"up": 0, "down": 0, "neutral": 0, "total": 0}
        
        regions[region]["total"] += 1
        if market["change"] > 0:
            regions[region]["up"] += 1
        elif market["change"] < 0:
            regions[region]["down"] += 1
        else:
            regions[region]["neutral"] += 1
    
    return {
        "total_markets": len(market_data),
        "up_markets": up_markets,
        "down_markets": down_markets,
        "neutral_markets": neutral_markets,
        "regions": regions,
        "market_sentiment": "bullish" if up_markets > down_markets else "bearish" if down_markets > up_markets else "mixed"
    }

@router.get("/market-overview")
def market_overview():
    try:
        market_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # fetch last 60 days (50 trading days)
        successful_fetches = 0
        total_attempts = len(MAJOR_MARKETS)

        for name, ticker in MAJOR_MARKETS.items():
            try:
                # Create yfinance ticker object
                stock = yf.Ticker(ticker)
                
                # Fetch historical data
                hist = stock.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1d"
                )

                if hist.empty or len(hist) < 2:  # Need at least 2 data points for change calculation
                    print(f"Insufficient data for {name} ({ticker})")
                    continue

                # Get latest and previous data for change calculation
                latest = hist.iloc[-1]
                previous = hist.iloc[-2]
                
                current_price = latest['Close']
                previous_close = previous['Close']
                volume = int(latest['Volume']) if pd.notna(latest['Volume']) else 0
                
                # Calculate actual changes
                change = round(current_price - previous_close, 2)
                percent_change = round((change / previous_close) * 100, 2) if previous_close != 0 else 0.0

                # Prepare history array for charting (last 30 days for cleaner charts)
                history_data = hist.tail(30)  # Last 30 data points for cleaner visualization
                history = [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "open": round(row['Open'], 2),
                        "high": round(row['High'], 2),
                        "low": round(row['Low'], 2),
                        "close": round(row['Close'], 2),
                        "volume": int(row['Volume']) if pd.notna(row['Volume']) else 0,
                    }
                    for date, row in history_data.iterrows()
                ]

                market_data.append({
                    "name": name,
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change": change,
                    "percent_change": percent_change,
                    "volume": volume,
                    "previous_close": round(previous_close, 2),
                    "currency": get_market_currency(ticker),
                    "region": get_market_region(ticker),
                    "last_updated": datetime.now().isoformat(),
                    "history": history
                })
                
                successful_fetches += 1

            except Exception as e:
                print(f"Error fetching {name} ({ticker}): {e}")
                continue

        print(f"Successfully fetched {successful_fetches}/{total_attempts} markets")

        # Calculate summary based on actual market changes
        summary = calculate_market_summary(market_data)

        return {
            "timestamp": datetime.now().isoformat(),
            "markets": market_data,
            "summary": summary,
            "metadata": {
                "successful_fetches": successful_fetches,
                "total_attempts": total_attempts,
                "success_rate": round((successful_fetches / total_attempts) * 100, 1) if total_attempts > 0 else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in global markets endpoint: {e}")
from fastapi import APIRouter, HTTPException
from polygon import RESTClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

POLY_KEY = os.getenv("POLYGON_API_KEY")
if not POLY_KEY:
    raise ValueError("POLYGON_API_KEY not set in environment variables")

client = RESTClient(api_key=POLY_KEY)

MAJOR_MARKETS = {
    "S&P 500": "SPY",   # ETF proxy
    "NASDAQ": "QQQ",    # ETF proxy
}

def calculate_market_summary(market_data):
    """Calculate market summary statistics based on actual market changes"""
    up_markets = sum(1 for m in market_data if m["change"] > 0)
    down_markets = sum(1 for m in market_data if m["change"] < 0)
    neutral_markets = sum(1 for m in market_data if m["change"] == 0)
    
    return {
        "total_markets": len(market_data),
        "up_markets": up_markets,
        "down_markets": down_markets,
        "neutral_markets": neutral_markets,
    }

@router.get("/market-overview")
def market_overview():
    try:
        market_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # fetch last 60 days (50 trading days)

        for name, ticker in MAJOR_MARKETS.items():
            try:
                # Historical OHLCV data
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    limit=50
                )

                if not aggs or len(aggs) < 2:  # Need at least 2 data points for change calculation
                    continue

                # Get latest and previous candles for change calculation
                latest = aggs[-1]
                previous = aggs[-2]
                
                current_price = latest.close
                previous_close = previous.close
                volume = latest.volume
                
                # Calculate actual changes
                change = round(current_price - previous_close, 2)
                percent_change = round((change / previous_close) * 100, 2) if previous_close != 0 else 0.0

                # Prepare history array for charting (last 30 days for cleaner charts)
                history = [
                    {
                        "date": datetime.fromtimestamp(a.timestamp / 1000).strftime("%Y-%m-%d"),
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                    }
                    for a in aggs[-30:]  # Last 30 data points for cleaner visualization
                ]

                market_data.append({
                    "name": name,
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change": change,
                    "percent_change": percent_change,
                    "volume": volume,
                    "previous_close": round(previous_close, 2),
                    "currency": "$",  # Changed to just "$" to match frontend
                    "last_updated": datetime.now().isoformat(),
                    "history": history
                })

            except Exception as e:
                print(f"Error fetching {name} ({ticker}): {e}")
                continue

        # Calculate summary based on actual market changes
        summary = calculate_market_summary(market_data)

        return {
            "timestamp": datetime.now().isoformat(),
            "markets": market_data,
            "summary": summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in polygon MVP endpoint: {e}")
from fastapi import APIRouter, HTTPException, Path, Query
import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, List, Optional
import time
from bs4 import BeautifulSoup
import re

from services.data_provider import get_data_provider

router = APIRouter()

# SEC EDGAR CIK mapping for major institutions
MAJOR_WHALES = {
    "Berkshire Hathaway": "0001067983",
    "Vanguard Group": "0000102909", 
    "BlackRock": "0001364742",
    "State Street": "0000093751",
    "Fidelity": "0000315066",
    "JPMorgan Chase": "0000019617",
    "Bank of America": "0000070858",
    "Wells Fargo": "0000072971",
    "Goldman Sachs": "0000886982",
    "Morgan Stanley": "0000895421"
}

class WhaleTracker:
    def __init__(self):
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
    
    def get_13f_filings(self, cik: str, limit: int = 5) -> List[Dict]:
        """Get recent 13F filings for a CIK"""
        try:
            # SEC EDGAR search URL
            search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={cik}&type=13F&dateb=&owner=include&count={limit}"
            
            response = requests.get(search_url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML to extract filing information
            soup = BeautifulSoup(response.content, 'html.parser')
            filings = []
            
            # Look for filing rows in the table
            rows = soup.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4 and '13F-HR' in row.get_text():
                    date_col = cols[3].get_text().strip()
                    link_col = cols[1].find('a')
                    
                    if link_col and date_col:
                        filing_url = "https://www.sec.gov" + link_col.get('href')
                        filings.append({
                            'date': date_col,
                            'url': filing_url,
                            'type': '13F-HR'
                        })
            
            return filings[:limit]
        except Exception as e:
            print(f"Error fetching 13F filings for CIK {cik}: {e}")
            return []

    def analyze_unusual_volume(self, symbol: str, days: int = 30) -> Dict:
        """Analyze unusual volume patterns that might indicate whale activity"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data_provider = get_data_provider()
            hist = data_provider.get_stock_history(
                symbol.upper(),
                start=start_date,
                end=end_date
            )

            if hist.empty:
                return {"error": "No data available"}
            
            # Calculate average volume and identify unusual spikes
            avg_volume = hist['Volume'].mean()
            volume_std = hist['Volume'].std()
            
            # Define unusual volume as 2+ standard deviations above average
            unusual_threshold = avg_volume + (2 * volume_std)
            
            unusual_days = hist[hist['Volume'] > unusual_threshold].copy()
            unusual_days['volume_ratio'] = unusual_days['Volume'] / avg_volume
            
            recent_volume = hist['Volume'].tail(5).mean()
            volume_trend = "increasing" if recent_volume > avg_volume * 1.2 else "decreasing" if recent_volume < avg_volume * 0.8 else "stable"
            
            return {
                "symbol": symbol,
                "average_volume": int(avg_volume),
                "recent_avg_volume": int(recent_volume),
                "unusual_threshold": int(unusual_threshold),
                "unusual_volume_days": len(unusual_days),
                "volume_trend": volume_trend,
                "unusual_days": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "volume": int(row['Volume']),
                        "volume_ratio": round(row['volume_ratio'], 2),
                        "price": round(row['Close'], 2),
                        "price_change_pct": round(((row['Close'] - row['Open']) / row['Open']) * 100, 2)
                    }
                    for date, row in unusual_days.tail(10).iterrows()
                ]
            }
        except Exception as e:
            return {"error": f"Error analyzing volume for {symbol}: {e}"}

    def get_insider_trades(self, symbol: str) -> List[Dict]:
        """Get recent insider trading data (using free sources)"""
        try:
            # Get company info
            data_provider = get_data_provider()
            info = data_provider.get_stock_info(symbol.upper())
            
            # This is a simplified version - in practice you'd want to scrape
            # from sources like OpenInsider or SEC EDGAR directly
            insider_data = []
            
            # Placeholder for actual insider data scraping
            # You would implement web scraping from OpenInsider.com here
            
            return {
                "symbol": symbol,
                "company_name": info.get('longName', 'Unknown'),
                "recent_insider_trades": insider_data,
                "note": "Insider data requires web scraping from OpenInsider or SEC EDGAR"
            }
        except Exception as e:
            return {"error": f"Error getting insider trades for {symbol}: {e}"}

    def detect_dark_pool_activity(self, symbol: str) -> Dict:
        """Detect potential dark pool activity through price/volume analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            data_provider = get_data_provider()
            hist = data_provider.get_stock_history(
                symbol.upper(),
                start=start_date,
                end=end_date
            )
            
            if hist.empty:
                return {"error": "No data available"}
            
            # Look for signs of dark pool activity:
            # 1. Price movements without proportional volume
            # 2. After-hours price gaps
            # 3. Consistent buying/selling pressure
            
            signals = []
            
            for i in range(1, len(hist)):
                current = hist.iloc[i]
                previous = hist.iloc[i-1]
                
                price_change_pct = ((current['Close'] - previous['Close']) / previous['Close']) * 100
                volume_ratio = current['Volume'] / hist['Volume'].mean()
                
                # Flag: Large price movement with below-average volume
                if abs(price_change_pct) > 2 and volume_ratio < 0.7:
                    signals.append({
                        "date": current.name.strftime("%Y-%m-%d"),
                        "type": "low_volume_price_movement",
                        "price_change_pct": round(price_change_pct, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "description": "Large price move with unusually low volume"
                    })
                
                # Flag: Price gap with normal volume (potential after-hours activity)
                gap = abs((current['Open'] - previous['Close']) / previous['Close']) * 100
                if gap > 1 and 0.8 <= volume_ratio <= 1.2:
                    signals.append({
                        "date": current.name.strftime("%Y-%m-%d"),
                        "type": "price_gap",
                        "gap_pct": round(gap, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "description": "Price gap suggesting after-hours institutional activity"
                    })
            
            return {
                "symbol": symbol,
                "analysis_period_days": 30,
                "dark_pool_signals": signals[-10:],  # Last 10 signals
                "total_signals": len(signals),
                "risk_level": "high" if len(signals) > 10 else "medium" if len(signals) > 5 else "low"
            }
            
        except Exception as e:
            return {"error": f"Error analyzing dark pool activity for {symbol}: {e}"}

whale_tracker = WhaleTracker()

@router.get("/whale-activity/{symbol}")
@router.get("/whale-activity/{symbol}")
def get_whale_activity(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL)")
):
    """Get comprehensive whale activity analysis for a symbol"""
    try:
        # Analyze unusual volume
        volume_analysis = whale_tracker.analyze_unusual_volume(symbol.upper())
        
        # Detect dark pool signals
        dark_pool_analysis = whale_tracker.detect_dark_pool_activity(symbol.upper())
        
        # Get basic stock info
        data_provider = get_data_provider()
        info = data_provider.get_stock_info(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "company_name": info.get('longName', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "volume_analysis": volume_analysis,
            "dark_pool_analysis": dark_pool_analysis,
            "market_cap": info.get('marketCap'),
            "avg_volume": info.get('averageVolume'),
            "institutional_ownership_pct": info.get('heldByInstitutions', 0) * 100 if info.get('heldByInstitutions') else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing whale activity: {e}")

@router.get("/institutional-filings")
def get_institutional_filings(limit: int = Query(10, description="Number of recent filings to fetch")):
    """Get recent 13F filings from major institutional investors"""
    try:
        all_filings = []
        
        for institution, cik in MAJOR_WHALES.items():
            filings = whale_tracker.get_13f_filings(cik, limit=3)
            for filing in filings:
                filing['institution'] = institution
                filing['cik'] = cik
                all_filings.append(filing)
            
            # Rate limiting to be respectful to SEC servers
            time.sleep(0.1)
        
        # Sort by date (most recent first)
        all_filings.sort(key=lambda x: x['date'], reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_filings": len(all_filings),
            "filings": all_filings[:limit],
            "institutions_tracked": list(MAJOR_WHALES.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching institutional filings: {e}")

@router.get("/unusual-volume-scanner")
def unusual_volume_scanner(
    symbols: str = Query(..., description="Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)"),
    min_volume_ratio: float = Query(2.0, description="Minimum volume ratio to flag as unusual")
):
    """Scan multiple symbols for unusual volume activity"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        results = []
        
        for symbol in symbol_list:
            analysis = whale_tracker.analyze_unusual_volume(symbol, days=30)
            
            if 'error' not in analysis:
                # Filter for significant unusual activity
                recent_unusual = [
                    day for day in analysis.get('unusual_days', [])
                    if day['volume_ratio'] >= min_volume_ratio
                ]
                
                if recent_unusual:
                    results.append({
                        "symbol": symbol,
                        "unusual_activity_days": len(recent_unusual),
                        "highest_volume_ratio": max(day['volume_ratio'] for day in recent_unusual),
                        "average_volume": analysis['average_volume'],
                        "volume_trend": analysis['volume_trend'],
                        "recent_unusual_days": recent_unusual[-3:]  # Last 3 unusual days
                    })
            
            time.sleep(0.1)  # Rate limiting
        
        # Sort by most unusual activity
        results.sort(key=lambda x: x['highest_volume_ratio'], reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols_scanned": len(symbol_list),
            "symbols_with_unusual_activity": len(results),
            "min_volume_ratio_threshold": min_volume_ratio,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning for unusual volume: {e}")

@router.get("/whale-watchlist")
def get_whale_watchlist():
    """Get a predefined watchlist of stocks commonly targeted by institutional investors"""
    try:
        # Common whale targets - large cap, high volume stocks
        watchlist_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B",
            "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE",
            "CRM", "NFLX", "XOM", "PFE", "TMO", "ABBV", "COST"
        ]
        
        whale_activity = []
        
        for symbol in watchlist_symbols[:10]:  # Limit to first 10 for performance
            try:
                data_provider = get_data_provider()
                hist = data_provider.get_stock_history(
                    symbol.upper(),
                    period="5d"
                )
                info = data_provider.get_stock_info(symbol.upper())

                if not hist.empty:
                    current_volume = hist['Volume'].iloc[-1]
                    avg_volume = info.get('averageVolume', 0)
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    
                    whale_activity.append({
                        "symbol": symbol,
                        "company_name": info.get('longName', 'Unknown'),
                        "current_price": round(hist['Close'].iloc[-1], 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "market_cap": info.get('marketCap'),
                        "institutional_ownership": round(info.get('heldByInstitutions', 0) * 100, 1) if info.get('heldByInstitutions') else None,
                        "whale_interest": "high" if volume_ratio > 1.5 else "medium" if volume_ratio > 1.2 else "low"
                    })
                
                time.sleep(0.05)  # Small delay to avoid rate limiting
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        return {
            "timestamp": datetime.now().isoformat(),
            "watchlist": whale_activity,
            "total_symbols": len(whale_activity),
            "high_interest_count": sum(1 for item in whale_activity if item['whale_interest'] == 'high')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating whale watchlist: {e}")
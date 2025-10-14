from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from data.tickers import STOCK_TICKERS
from services.data_provider import get_data_provider

router = APIRouter()

# Stock Assessment Configuration
ASSESSMENT_CONFIG = {
    "metrics": {
        # Valuation Metrics
        "pe_ratio": {"weight": 0.20, "optimal_range": (10, 25)},
        "pb_ratio": {"weight": 0.15, "optimal_range": (1, 3)},
        "peg_ratio": {"weight": 0.15, "optimal_range": (0.5, 1.5)},
        
        # Profitability Metrics  
        "roe": {"weight": 0.15, "optimal_range": (15, 30)},
        "roa": {"weight": 0.10, "optimal_range": (5, 15)},
        "profit_margin": {"weight": 0.10, "optimal_range": (10, 25)},
        
        # Growth Metrics
        "revenue_growth": {"weight": 0.10, "optimal_range": (5, 20)},
        "earnings_growth": {"weight": 0.10, "optimal_range": (5, 25)},
        
        # Financial Health
        "debt_to_equity": {"weight": 0.08, "optimal_range": (0, 0.5)},
        "current_ratio": {"weight": 0.07, "optimal_range": (1.5, 3)},
    },
    "tiers": {
        "excellent": {"min_score": 80, "color": "#10b981", "description": "High-quality stocks with strong fundamentals"},
        "good": {"min_score": 60, "color": "#3b82f6", "description": "Solid stocks with decent fundamentals"},
        "average": {"min_score": 40, "color": "#f59e0b", "description": "Average stocks with mixed fundamentals"},
        "poor": {"min_score": 20, "color": "#ef4444", "description": "Below-average stocks with weak fundamentals"},
        "avoid": {"min_score": 0, "color": "#991b1b", "description": "High-risk stocks with poor fundamentals"}
    }
}

def calculate_metric_score(value: float, metric_config: Dict[str, Any]) -> float:
    """Calculate a 0-100 score for a metric based on its optimal range"""
    if value is None or value == 0:
        return 0
    
    optimal_min, optimal_max = metric_config["optimal_range"]
    
    # Handle metrics where lower is better (like debt-to-equity, PE ratio)
    if metric_config.get("lower_is_better", False):
        if value <= optimal_min:
            return 100
        elif value <= optimal_max:
            return 100 - ((value - optimal_min) / (optimal_max - optimal_min)) * 50
        else:
            return max(0, 50 - ((value - optimal_max) / optimal_max) * 50)
    
    # Handle metrics where higher is better (like ROE, growth rates)
    else:
        if optimal_min <= value <= optimal_max:
            return 100
        elif value < optimal_min:
            return max(0, (value / optimal_min) * 100)
        else:
            # Diminishing returns for values above optimal range
            excess = (value - optimal_max) / optimal_max
            return max(50, 100 - (excess * 25))

def get_stock_fundamentals(ticker: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive stock fundamentals"""
    try:
        data_provider = get_data_provider()
        info = data_provider.get_stock_info(ticker)
        
        # Extract key metrics (handle missing data gracefully)
        fundamentals = {
            # Basic Info
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            
            # Valuation Metrics
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "peg_ratio": info.get("pegRatio"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_revenue": info.get("enterpriseToRevenue"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            
            # Profitability Metrics
            "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
            "roa": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else None,
            "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
            "operating_margin": info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else None,
            "gross_margin": info.get("grossMargins", 0) * 100 if info.get("grossMargins") else None,
            
            # Growth Metrics
            "revenue_growth": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
            "earnings_growth": info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else None,
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth", 0) * 100 if info.get("earningsQuarterlyGrowth") else None,
            
            # Financial Health
            "debt_to_equity": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None,
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_assets": info.get("totalDebt", 0) / max(info.get("totalAssets", 1), 1),
            
            # Dividend Metrics
            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "payout_ratio": info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else None,
            "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
            
            # Price and Volume Data
            "current_price": info.get("regularMarketPrice") or info.get("previousClose", 0),
            "previous_close": info.get("regularMarketPreviousClose", 0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "average_volume": info.get("averageVolume"),
            
            # Additional Metrics
            "beta": info.get("beta"),
            "book_value": info.get("bookValue"),
            "price_to_book": info.get("priceToBook"),
            "forward_pe": info.get("forwardPE"),
            "trailing_annual_dividend_rate": info.get("trailingAnnualDividendRate"),
        }
        
        return fundamentals
        
    except Exception as e:
        print(f"Error fetching fundamentals for {ticker}: {e}")
        return None

def calculate_stock_score(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall stock quality score based on fundamentals"""
    metrics = ASSESSMENT_CONFIG["metrics"]
    scores = {}
    weighted_score = 0
    total_weight = 0
    
    for metric_name, config in metrics.items():
        value = fundamentals.get(metric_name)
        if value is not None:
            score = calculate_metric_score(value, config)
            scores[metric_name] = {
                "value": round(value, 2),
                "score": round(score, 1),
                "weight": config["weight"]
            }
            weighted_score += score * config["weight"]
            total_weight += config["weight"]
    
    # Calculate final score (0-100)
    final_score = (weighted_score / total_weight) if total_weight > 0 else 0
    
    # Determine quality tier
    tier = "avoid"
    for tier_name, tier_config in ASSESSMENT_CONFIG["tiers"].items():
        if final_score >= tier_config["min_score"]:
            tier = tier_name
            break
    
    return {
        "overall_score": round(final_score, 1),
        "tier": tier,
        "tier_info": ASSESSMENT_CONFIG["tiers"][tier],
        "metric_scores": scores,
        "metrics_evaluated": len(scores),
        "total_possible_metrics": len(metrics)
    }

def get_price_history(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get price history for charting"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        data_provider = get_data_provider()
        hist = data_provider.get_stock_history(ticker, start=start_date, end=end_date)

        if hist.empty:
            return []
            
        return [
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume']),
            }
            for date, row in hist.tail(days).iterrows()
        ]
    except Exception as e:
        print(f"Error fetching price history for {ticker}: {e}")
        return []

@router.get("/stock-assessment")
def get_stock_assessment():
    """Get comprehensive stock assessment with quality tiers"""
    try:
        assessments = []
        
        for ticker in STOCK_TICKERS:
            print(f"Assessing {ticker}...")
            
            # Get fundamental data
            fundamentals = get_stock_fundamentals(ticker)
            if not fundamentals:
                continue
            
            # Calculate quality score and tier
            assessment = calculate_stock_score(fundamentals)
            
            # Get price history
            price_history = get_price_history(ticker)
            
            # Calculate price change
            current_price = fundamentals["current_price"]
            previous_close = fundamentals["previous_close"]
            price_change = current_price - previous_close if current_price and previous_close else 0
            price_change_percent = (price_change / previous_close * 100) if previous_close else 0
            
            stock_assessment = {
                "ticker": ticker,
                "name": fundamentals["name"],
                "sector": fundamentals["sector"],
                "industry": fundamentals["industry"],
                "current_price": round(current_price, 2) if current_price else 0,
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_percent, 2),
                "market_cap": fundamentals["market_cap"],
                
                # Assessment Results
                "overall_score": assessment["overall_score"],
                "tier": assessment["tier"],
                "tier_info": assessment["tier_info"],
                
                # Key Fundamentals (for quick reference)
                "key_metrics": {
                    "pe_ratio": fundamentals["pe_ratio"],
                    "pb_ratio": fundamentals["pb_ratio"],
                    "roe": fundamentals["roe"],
                    "debt_to_equity": fundamentals["debt_to_equity"],
                    "dividend_yield": fundamentals["dividend_yield"],
                    "beta": fundamentals["beta"]
                },
                
                # Detailed Assessment
                "detailed_assessment": {
                    "metric_scores": assessment["metric_scores"],
                    "metrics_evaluated": assessment["metrics_evaluated"],
                    "total_possible_metrics": assessment["total_possible_metrics"]
                },
                
                # Full Fundamentals (for detailed analysis)
                "fundamentals": fundamentals,
                "price_history": price_history,
                "last_updated": datetime.now().isoformat(),
            }
            
            assessments.append(stock_assessment)
        
        # Define tier order (best to worst)
        tier_order = ["excellent", "good", "average", "poor", "avoid"]
        
        # Group by tiers using ordered dictionary
        from collections import OrderedDict
        tiers = OrderedDict()
        for tier in tier_order:
            tiers[tier] = []
        
        for assessment in assessments:
            tier = assessment["tier"]
            tiers[tier].append(assessment)
        
        # Sort each tier by score (highest first)
        for tier_stocks in tiers.values():
            tier_stocks.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Remove empty tiers for cleaner output (optional)
        tiers = OrderedDict((tier, stocks) for tier, stocks in tiers.items() if stocks)
        
        # Calculate summary statistics
        total_stocks = len(assessments)
        avg_score = sum(a["overall_score"] for a in assessments) / total_stocks if total_stocks > 0 else 0
        
        tier_counts = {tier: len(stocks) for tier, stocks in tiers.items()}
        
        # Sector analysis
        sectors = {}
        for assessment in assessments:
            sector = assessment["sector"]
            if sector not in sectors:
                sectors[sector] = {"count": 0, "avg_score": 0, "total_score": 0}
            sectors[sector]["count"] += 1
            sectors[sector]["total_score"] += assessment["overall_score"]
        
        for sector_data in sectors.values():
            sector_data["avg_score"] = round(sector_data["total_score"] / sector_data["count"], 1)
            del sector_data["total_score"]  # Remove intermediate calculation
        
        return {
            "timestamp": datetime.now().isoformat(),
            "assessments": assessments,
            "tiers": tiers,
            "summary": {
                "total_stocks_assessed": total_stocks,
                "average_score": round(avg_score, 1),
                "tier_distribution": tier_counts,
                "sector_analysis": sectors,
                "assessment_config": ASSESSMENT_CONFIG
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock assessment error: {e}")
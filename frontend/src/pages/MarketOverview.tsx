// src/pages/MarketOverview.tsx
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface MarketHistory {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Market {
  name: string;
  ticker: string;
  price: number;
  change: number;
  percent_change: number;
  volume: number;
  last_updated: string;
  previous_close: number;
  currency: string;
  region: string;
  history: MarketHistory[];
}

interface RegionSummary {
  up: number;
  down: number;
  neutral: number;
  total: number;
}

interface MarketOverviewResponse {
  timestamp: string;
  markets: Market[];
  summary: {
    total_markets: number;
    up_markets: number;
    down_markets: number;
    neutral_markets: number;
    regions: Record<string, RegionSummary>;
    market_sentiment: string;
  };
  metadata: {
    successful_fetches: number;
    total_attempts: number;
    success_rate: number;
  };
}

// Helper function to get region emoji
const getRegionEmoji = (region: string): string => {
  const emojiMap: Record<string, string> = {
    "North America": "🇺🇸",
    "Europe": "🇪🇺", 
    "Asia-Pacific": "🌏",
    "Other": "🌍"
  };
  return emojiMap[region] || "🌍";
};

// Helper function to get sentiment color
const getSentimentColor = (sentiment: string): string => {
  switch (sentiment) {
    case "bullish": return "text-green-600";
    case "bearish": return "text-red-600";
    default: return "text-yellow-600";
  }
};

export default function MarketOverview() {
  const [expandedRegions, setExpandedRegions] = useState<Record<string, boolean>>({
    "North America": true, // Start with North America expanded
    "Europe": false,
    "Asia-Pacific": false,
    "Other": false
  });

  const { data, isLoading, error } = useQuery<MarketOverviewResponse>({
    queryKey: ["market-overview"],
    queryFn: async () => {
      const res = await axios.get("http://127.0.0.1:8000/api/market-overview");
      return res.data;
    },
    retry: 1,
    retryDelay: 10000, // 10 seconds
    staleTime: 30 * 1000, // cache lifetime
    refetchInterval: 30 * 1000, // only refetch every 30s
    refetchOnWindowFocus: false, // prevent extra calls
    refetchIntervalInBackground: false, // optional
  });

  const toggleRegion = (region: string) => {
    setExpandedRegions(prev => ({
      ...prev,
      [region]: !prev[region]
    }));
  };

  const expandAllRegions = () => {
    const allExpanded = Object.values(expandedRegions).every(Boolean);
    const newState = Object.keys(expandedRegions).reduce((acc, region) => ({
      ...acc,
      [region]: !allExpanded
    }), {});
    setExpandedRegions(newState);
  };

  if (isLoading) return <div className="p-6">Loading market data...</div>;
  if (error) return <div className="p-6 text-red-500">Error loading market data</div>;
  if (!data) return <div className="p-6">No market data available</div>;

  // Group markets by region
  const marketsByRegion = data.markets.reduce((acc, market) => {
    const region = market.region || "Other";
    if (!acc[region]) acc[region] = [];
    acc[region].push(market);
    return acc;
  }, {} as Record<string, Market[]>);

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Global Market Overview</h1>
        <div className="flex items-center space-x-4">
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            data.summary.market_sentiment === 'bullish' ? 'bg-green-100 text-green-800' :
            data.summary.market_sentiment === 'bearish' ? 'bg-red-100 text-red-800' :
            'bg-yellow-100 text-yellow-800'
          }`}>
            Market Sentiment: {data.summary.market_sentiment.charAt(0).toUpperCase() + data.summary.market_sentiment.slice(1)}
          </div>
          <button
            onClick={expandAllRegions}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
          >
            {Object.values(expandedRegions).every(Boolean) ? "Collapse All" : "Expand All"}
          </button>
        </div>
      </div>
      
      {/* Global Summary Section */}
      <div className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border">
        <h2 className="text-xl font-semibold mb-4">Global Market Summary</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center mb-4">
          <div>
            <div className="text-2xl font-bold text-blue-600">{data.summary.total_markets}</div>
            <div className="text-sm text-gray-600">Total Markets</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">{data.summary.up_markets}</div>
            <div className="text-sm text-gray-600">Markets Up</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-600">{data.summary.down_markets}</div>
            <div className="text-sm text-gray-600">Markets Down</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-600">{data.summary.neutral_markets}</div>
            <div className="text-sm text-gray-600">Neutral</div>
          </div>
        </div>
        
        {/* Data Quality Indicator */}
        <div className="text-sm text-gray-600 text-center">
          Successfully loaded {data.metadata.successful_fetches}/{data.metadata.total_attempts} markets 
          ({data.metadata.success_rate}% success rate)
        </div>
      </div>

      {/* Regional Panels */}
      <div className="space-y-6">
        {Object.entries(marketsByRegion).map(([region, markets]) => {
          const regionSummary = data.summary.regions[region];
          const isExpanded = expandedRegions[region];
          
          return (
            <div key={region} className="border rounded-lg shadow-sm bg-white">
              {/* Region Header */}
              <div 
                className="p-4 bg-gray-50 rounded-t-lg cursor-pointer hover:bg-gray-100 transition-colors"
                onClick={() => toggleRegion(region)}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{getRegionEmoji(region)}</span>
                    <h2 className="text-xl font-semibold">{region}</h2>
                    <span className="text-sm text-gray-500">({markets.length} markets)</span>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    {/* Region Summary Stats */}
                    {regionSummary && (
                      <div className="flex items-center space-x-4 text-sm">
                        <span className="text-green-600 font-medium">↑{regionSummary.up}</span>
                        <span className="text-red-600 font-medium">↓{regionSummary.down}</span>
                        <span className="text-gray-600 font-medium">−{regionSummary.neutral}</span>
                      </div>
                    )}
                    
                    {/* Expand/Collapse Arrow */}
                    <svg 
                      className={`w-5 h-5 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </div>

              {/* Region Content */}
              {isExpanded && (
                <div className="p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {markets.map((market) => (
                      <div key={market.ticker} className="p-4 border rounded-lg shadow-sm bg-white hover:shadow-md transition-shadow">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <h3 className="font-semibold text-lg">{market.name}</h3>
                            <p className="text-sm text-gray-500">{market.ticker}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-xl font-bold">
                              {market.currency === "USD" ? "$" : market.currency + " "}{market.price.toLocaleString()}
                            </div>
                            <div className={`text-sm ${market.change >= 0 ? "text-green-600" : "text-red-600"}`}>
                              {market.change >= 0 ? "+" : ""}
                              {market.change.toFixed(2)} ({market.percent_change.toFixed(2)}%)
                            </div>
                          </div>
                        </div>

                        {/* Chart */}
                        <div className="w-full h-[200px] mb-2">
                          <ResponsiveContainer width="100%" height={200}>
                            <LineChart 
                              data={market.history}
                              margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                            >
                              <XAxis dataKey="date" hide />
                              <YAxis hide domain={["dataMin - 1", "dataMax + 1"]} />
                              <Tooltip 
                                formatter={(value, name) => {
                                  const numValue = typeof value === "number" ? value : Number(value);
                                  const currencySymbol = market.currency === "USD" ? "$" : market.currency + " ";
                                  return [`${currencySymbol}${numValue.toFixed(2)}`, 'Close'];
                                }}
                                labelFormatter={(label) => `Date: ${label}`}
                              />
                              <Line
                                type="monotone"
                                dataKey="close"
                                stroke={market.change >= 0 ? "#16a34a" : market.change < 0 ? "#dc2626" : "#6b7280"}
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4 }}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="text-xs text-gray-500 space-y-1">
                          <div>Volume: {market.volume.toLocaleString()}</div>
                          <div>Previous Close: {market.currency === "USD" ? "$" : market.currency + " "}{market.previous_close.toFixed(2)}</div>
                          <div>Last Updated: {new Date(market.last_updated).toLocaleTimeString()}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-6 text-sm text-gray-500 text-center">
        Data last updated: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
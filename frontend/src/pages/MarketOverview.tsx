// src/pages/MarketOverview.tsx
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

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
}

interface MarketOverviewResponse {
  timestamp: string;
  markets: Market[];
  summary: {
    total_markets: number;
    up_markets: number;
    down_markets: number;
    neutral_markets: number;
  };
}

export default function MarketOverview() {
  const { data, isLoading, error } = useQuery<MarketOverviewResponse>({
    queryKey: ["market-overview"],
    queryFn: async () => {
      console.log("Making API request to backend...");
      try {
        const res = await axios.get("http://127.0.0.1:8000/api/market-overview");
        console.log("API response:", res.data);
        return res.data;
      } catch (err) {
        console.error("API request failed:", err);
        throw err;
      }
    },
    retry: 1,
    retryDelay: 1000,
  });

  console.log("Query state:", { isLoading, error, data });

  if (isLoading) return <div className="p-6">Loading market data...</div>;
  if (error) {
    console.error("Query error:", error);
    return <div className="p-6 text-red-500">Error loading market data: {error instanceof Error ? error.message : 'Unknown error'}</div>;
  }

  if (!data || !data.markets) {
    return <div className="p-6">No market data available</div>;
  }

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Market Overview</h1>
      
      {/* Summary Section */}
      <div className="mb-8 p-4 bg-gray-100 rounded-lg">
        <h2 className="text-xl font-semibold mb-2">Market Summary</h2>
        <div className="grid grid-cols-4 gap-4 text-center">
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
      </div>

      {/* Markets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.markets.map((market) => (
          <div key={market.ticker} className="p-4 border rounded-lg shadow-sm bg-white">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="font-semibold text-lg">{market.name}</h3>
                <p className="text-sm text-gray-500">{market.ticker}</p>
              </div>
              <div className="text-right">
                <div className="text-xl font-bold">${market.price.toLocaleString()}</div>
                <div className={`text-sm ${
                  market.change >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {market.change >= 0 ? '+' : ''}{market.change.toFixed(2)} ({market.percent_change.toFixed(2)}%)
                </div>
              </div>
            </div>
            <div className="text-xs text-gray-500 space-y-1">
              <div>Volume: {market.volume.toLocaleString()}</div>
              <div>Previous Close: ${market.previous_close.toFixed(2)}</div>
              <div>Last Updated: {new Date(market.last_updated).toLocaleTimeString()}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-sm text-gray-500">
        Data last updated: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}

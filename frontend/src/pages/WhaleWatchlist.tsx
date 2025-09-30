import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

interface WhaleWatchlistItem {
  symbol: string;
  company_name: string;
  current_price: number;
  volume_ratio: number;
  market_cap: number;
  institutional_ownership: number | null;
  whale_interest: "high" | "medium" | "low";
}

interface WhaleWatchlistResponse {
  timestamp: string;
  watchlist: WhaleWatchlistItem[];
  total_symbols: number;
  high_interest_count: number;
}

// Helper function to get interest level styling
const getInterestLevelColor = (level: string) => {
  switch (level) {
    case "high": return "bg-red-100 text-red-800";
    case "medium": return "bg-yellow-100 text-yellow-800";
    case "low": return "bg-green-100 text-green-800";
    default: return "bg-gray-100 text-gray-800";
  }
};

const getInterestEmoji = (level: string) => {
  switch (level) {
    case "high": return "🔴";
    case "medium": return "🟡";
    case "low": return "🟢";
    default: return "⚪";
  }
};

export default function WhaleWatchlist() {
  const [sortBy, setSortBy] = useState<"volume_ratio" | "market_cap" | "institutional_ownership">("volume_ratio");
  const [filterLevel, setFilterLevel] = useState<"all" | "high" | "medium" | "low">("all");

  const { data, isLoading, error, refetch } = useQuery<WhaleWatchlistResponse>({
    queryKey: ["whale-watchlist"],
    queryFn: async () => {
      const res = await axios.get("http://127.0.0.1:8000/api/whale-watchlist");
      return res.data;
    },
    retry: 1,
    retryDelay: 10000,
    staleTime: 60 * 1000, // cache for 1 minute
    refetchInterval: 60 * 1000, // refetch every minute
    refetchOnWindowFocus: false,
    refetchIntervalInBackground: false,
  });

  if (isLoading) return <div className="p-6">Loading whale watchlist...</div>;
  if (error) return <div className="p-6 text-red-500">Error loading whale data</div>;
  if (!data) return <div className="p-6">No whale data available</div>;

  // Filter and sort data
  let filteredData = data.watchlist;
  if (filterLevel !== "all") {
    filteredData = filteredData.filter(item => item.whale_interest === filterLevel);
  }

  filteredData.sort((a, b) => {
    switch (sortBy) {
      case "volume_ratio":
        return b.volume_ratio - a.volume_ratio;
      case "market_cap":
        return (b.market_cap || 0) - (a.market_cap || 0);
      case "institutional_ownership":
        return (b.institutional_ownership || 0) - (a.institutional_ownership || 0);
      default:
        return 0;
    }
  });

  // Prepare chart data
  const interestDistribution = [
    { name: "High Interest", value: data.watchlist.filter(item => item.whale_interest === "high").length, color: "#dc2626" },
    { name: "Medium Interest", value: data.watchlist.filter(item => item.whale_interest === "medium").length, color: "#d97706" },
    { name: "Low Interest", value: data.watchlist.filter(item => item.whale_interest === "low").length, color: "#16a34a" },
  ];

  const volumeRatioData = data.watchlist
    .sort((a, b) => b.volume_ratio - a.volume_ratio)
    .slice(0, 10)
    .map(item => ({
      symbol: item.symbol,
      volume_ratio: item.volume_ratio,
      whale_interest: item.whale_interest
    }));

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">🐋 Whale Watchlist</h1>
        <div className="flex items-center space-x-4">
          <div className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
            {data.high_interest_count} High Interest Stocks
          </div>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
          >
            Refresh Data
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border">
          <h3 className="text-lg font-semibold mb-2">Total Tracked</h3>
          <div className="text-3xl font-bold text-blue-600">{data.total_symbols}</div>
          <div className="text-sm text-gray-600">Stocks monitored</div>
        </div>
        
        <div className="p-6 bg-gradient-to-r from-red-50 to-pink-50 rounded-lg border">
          <h3 className="text-lg font-semibold mb-2">High Interest</h3>
          <div className="text-3xl font-bold text-red-600">{data.high_interest_count}</div>
          <div className="text-sm text-gray-600">Showing whale activity</div>
        </div>
        
        <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border">
          <h3 className="text-lg font-semibold mb-2">Avg Volume Ratio</h3>
          <div className="text-3xl font-bold text-green-600">
            {(data.watchlist.reduce((sum, item) => sum + item.volume_ratio, 0) / data.watchlist.length).toFixed(1)}x
          </div>
          <div className="text-sm text-gray-600">Above normal volume</div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Interest Distribution Pie Chart */}
        <div className="p-6 bg-white rounded-lg border shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Whale Interest Distribution</h3>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={interestDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {interestDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value} stocks`, 'Count']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Volume Ratio Bar Chart */}
        <div className="p-6 bg-white rounded-lg border shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Top Volume Ratios</h3>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={volumeRatioData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                <XAxis dataKey="symbol" />
                <YAxis />
                <Tooltip 
                  formatter={(value) => [`${Number(value).toFixed(2)}x`, 'Volume Ratio']}
                  labelFormatter={(label) => `Symbol: ${label}`}
                />
                <Bar dataKey="volume_ratio">
                  {volumeRatioData.map((entry, index) => (
                    <Cell
                      key={`cell-bar-${index}`}
                      fill={
                        entry.whale_interest === 'high' ? '#dc2626' :
                        entry.whale_interest === 'medium' ? '#d97706' : '#16a34a'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-2">
          <label htmlFor="sortBySelect" className="text-sm font-medium">Sort by:</label>
          <select
            id="sortBySelect"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 border rounded-lg text-sm"
          >
            <option value="volume_ratio">Volume Ratio</option>
            <option value="market_cap">Market Cap</option>
            <option value="institutional_ownership">Institutional Ownership</option>
          </select>
        </div>
        
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium">Filter:</label>
          <label htmlFor="filterLevelSelect" className="sr-only">Whale Interest Level</label>
          <select
            id="filterLevelSelect"
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value as any)}
            className="px-3 py-1 border rounded-lg text-sm"
          >
            <option value="all">All Levels</option>
            <option value="high">High Interest Only</option>
            <option value="medium">Medium Interest Only</option>
            <option value="low">Low Interest Only</option>
          </select>
        </div>
      </div>

      {/* Watchlist Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredData.map((stock) => (
          <div key={stock.symbol} className="p-4 border rounded-lg shadow-sm bg-white hover:shadow-md transition-shadow">
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="font-bold text-lg">{stock.symbol}</h3>
                  <span className="text-xl">{getInterestEmoji(stock.whale_interest)}</span>
                </div>
                <p className="text-sm text-gray-600 truncate" title={stock.company_name}>
                  {stock.company_name}
                </p>
              </div>
              <div className="text-right">
                <div className="text-lg font-bold">${stock.current_price.toLocaleString()}</div>
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${getInterestLevelColor(stock.whale_interest)}`}>
                  {stock.whale_interest.toUpperCase()}
                </div>
              </div>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Volume Ratio:</span>
                <span className={`font-medium ${stock.volume_ratio > 1.5 ? 'text-red-600' : stock.volume_ratio > 1.2 ? 'text-yellow-600' : 'text-green-600'}`}>
                  {stock.volume_ratio.toFixed(2)}x
                </span>
              </div>
              
              {stock.market_cap && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Market Cap:</span>
                  <span className="font-medium">
                    ${(stock.market_cap / 1e9).toFixed(1)}B
                  </span>
                </div>
              )}
              
              {stock.institutional_ownership && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Institutional:</span>
                  <span className="font-medium">{stock.institutional_ownership.toFixed(1)}%</span>
                </div>
              )}
            </div>

            {/* Volume Ratio Progress Bar */}
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-600 mb-1">
                <span>Volume Activity</span>
                <span>{stock.volume_ratio.toFixed(2)}x normal</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    stock.volume_ratio > 2 ? 'bg-red-500' : 
                    stock.volume_ratio > 1.5 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${Math.min((stock.volume_ratio / 3) * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-sm text-gray-500 text-center">
        Data last updated: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
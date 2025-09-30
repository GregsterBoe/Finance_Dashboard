import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useState, useRef, type KeyboardEvent } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
} from "recharts";

interface UnusualDay {
  date: string;
  volume: number;
  volume_ratio: number;
  price: number;
  price_change_pct: number;
}

interface VolumeAnalysis {
  symbol: string;
  average_volume: number;
  recent_avg_volume: number;
  unusual_threshold: number;
  unusual_volume_days: number;
  volume_trend: "increasing" | "decreasing" | "stable";
  unusual_days: UnusualDay[];
}

interface DarkPoolSignal {
  date: string;
  type: "low_volume_price_movement" | "price_gap";
  price_change_pct?: number;
  gap_pct?: number;
  volume_ratio: number;
  description: string;
}

interface DarkPoolAnalysis {
  symbol: string;
  analysis_period_days: number;
  dark_pool_signals: DarkPoolSignal[];
  total_signals: number;
  risk_level: "high" | "medium" | "low";
}

interface WhaleActivityResponse {
  symbol: string;
  company_name: string;
  timestamp: string;
  volume_analysis: VolumeAnalysis;
  dark_pool_analysis: DarkPoolAnalysis;
  market_cap: number | null;
  avg_volume: number | null;
  institutional_ownership_pct: number | null;
}

export default function ActivityTracker() {
  const [symbol, setSymbol] = useState<string>("");
  const [searchSymbol, setSearchSymbol] = useState<string>("AAPL"); // Default symbol
  const [activeTab, setActiveTab] = useState<"volume" | "darkpool" | "overview">("overview");
  const inputRef = useRef<HTMLInputElement>(null);

  const { data, isLoading, error, refetch } = useQuery<WhaleActivityResponse>({
    queryKey: ["whale-activity", searchSymbol],
    queryFn: async () => {
      const res = await axios.get(`http://127.0.0.1:8000/api/whale-activity/${searchSymbol}`);
      return res.data;
    },
    enabled: !!searchSymbol,
    retry: 1,
    retryDelay: 5000,
    staleTime: 30 * 1000,
    refetchOnWindowFocus: false,
  });

  const handleSearch = () => {
    if (symbol.trim()) {
      setSearchSymbol(symbol.trim().toUpperCase());
      setSymbol("");
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "high": return "bg-red-100 text-red-800 border-red-200";
      case "medium": return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low": return "bg-green-100 text-green-800 border-green-200";
      default: return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "increasing": return "📈";
      case "decreasing": return "📉";
      case "stable": return "➡️";
      default: return "➡️";
    }
  };

  return (
    <div className="p-6">
      {/* Header with Search */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-4">🔍 Whale Activity Tracker</h1>
        
        {/* Search Section */}
        <div className="flex items-center space-x-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border">
          <div className="flex-1 max-w-md">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Stock Symbol
            </label>
            <div className="flex">
              <input
                ref={inputRef}
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                onKeyPress={handleKeyPress}
                placeholder="Enter symbol (e.g., AAPL, MSFT, TSLA)"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleSearch}
                className="px-6 py-2 bg-blue-500 text-white rounded-r-lg hover:bg-blue-600 transition-colors"
              >
                Search
              </button>
            </div>
          </div>
          
          {searchSymbol && (
            <div className="text-right">
              <div className="text-sm text-gray-600">Currently analyzing</div>
              <div className="text-xl font-bold text-blue-600">{searchSymbol}</div>
            </div>
          )}
        </div>
      </div>

      {isLoading && (
        <div className="flex justify-center items-center py-12">
          <div className="text-lg">Analyzing whale activity for {searchSymbol}...</div>
        </div>
      )}

      {error && (
        <div className="p-6 bg-red-50 border border-red-200 rounded-lg text-red-700">
          Error loading whale activity data. Please try a different symbol or check your connection.
        </div>
      )}

      {data && (
        <>
          {/* Company Info Header */}
          <div className="mb-6 p-6 bg-white rounded-lg border shadow-sm">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-2xl font-bold">{data.symbol}</h2>
                <p className="text-gray-600">{data.company_name}</p>
              </div>
              <div className="text-right">
                {data.market_cap && (
                  <div className="text-sm text-gray-600">
                    Market Cap: <span className="font-medium">${(data.market_cap / 1e9).toFixed(1)}B</span>
                  </div>
                )}
                {data.institutional_ownership_pct && (
                  <div className="text-sm text-gray-600">
                    Institutional: <span className="font-medium">{data.institutional_ownership_pct.toFixed(1)}%</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Quick Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-white rounded-lg border shadow-sm text-center">
              <div className="text-2xl font-bold text-blue-600">
                {data.volume_analysis.unusual_volume_days}
              </div>
              <div className="text-sm text-gray-600">Unusual Volume Days</div>
            </div>
            
            <div className="p-4 bg-white rounded-lg border shadow-sm text-center">
              <div className="text-2xl font-bold text-purple-600">
                {data.dark_pool_analysis.total_signals}
              </div>
              <div className="text-sm text-gray-600">Dark Pool Signals</div>
            </div>
            
            <div className="p-4 bg-white rounded-lg border shadow-sm text-center">
              <div className="text-2xl font-bold text-orange-600">
                {(data.volume_analysis.recent_avg_volume / data.volume_analysis.average_volume).toFixed(2)}x
              </div>
              <div className="text-sm text-gray-600">Recent Volume Ratio</div>
            </div>
            
            <div className={`p-4 rounded-lg border shadow-sm text-center ${getRiskLevelColor(data.dark_pool_analysis.risk_level)}`}>
              <div className="text-2xl font-bold">
                {data.dark_pool_analysis.risk_level.toUpperCase()}
              </div>
              <div className="text-sm">Whale Risk Level</div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="mb-6">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex space-x-8">
                {[
                  { id: "overview", label: "Overview", icon: "📊" },
                  { id: "volume", label: "Volume Analysis", icon: "📈" },
                  { id: "darkpool", label: "Dark Pool Signals", icon: "🕳️" }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? "border-blue-500 text-blue-600"
                        : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                    }`}
                  >
                    <span>{tab.icon}</span>
                    <span>{tab.label}</span>
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Tab Content */}
          {activeTab === "overview" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Volume Trend */}
              <div className="p-6 bg-white rounded-lg border shadow-sm">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <span className="mr-2">{getTrendIcon(data.volume_analysis.volume_trend)}</span>
                  Volume Trend Analysis
                </h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span>Average Volume:</span>
                    <span className="font-medium">{data.volume_analysis.average_volume.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Recent Average:</span>
                    <span className="font-medium">{data.volume_analysis.recent_avg_volume.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Unusual Threshold:</span>
                    <span className="font-medium">{data.volume_analysis.unusual_threshold.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Volume Trend:</span>
                    <span className={`font-medium px-2 py-1 rounded text-sm ${
                      data.volume_analysis.volume_trend === 'increasing' ? 'bg-red-100 text-red-800' :
                      data.volume_analysis.volume_trend === 'decreasing' ? 'bg-green-100 text-green-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {data.volume_analysis.volume_trend.toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Recent Unusual Days Chart */}
              <div className="p-6 bg-white rounded-lg border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Recent Unusual Volume Days</h3>
                <div className="w-full h-[250px]">
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={data.volume_analysis.unusual_days.slice(-10)} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <XAxis 
                        dataKey="date" 
                        tick={{ fontSize: 12 }}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip 
                        formatter={(value, name) => [
                          name === 'volume_ratio' ? `${Number(value).toFixed(2)}x` : Number(value).toLocaleString(),
                          name === 'volume_ratio' ? 'Volume Ratio' : 'Volume'
                        ]}
                      />
                      <Bar dataKey="volume_ratio" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === "volume" && (
            <div className="space-y-6">
              {/* Volume Analysis Details */}
              <div className="p-6 bg-white rounded-lg border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Detailed Volume Analysis</h3>
                
                {data.volume_analysis.unusual_days.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full">
                      <thead>
                        <tr className="bg-gray-50">
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Date</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Volume</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Volume Ratio</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Price</th>
                          <th className="px-4 py-2 text-left text-sm font-medium text-gray-700">Price Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.volume_analysis.unusual_days.map((day, index) => (
                          <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                            <td className="px-4 py-2 text-sm">{day.date}</td>
                            <td className="px-4 py-2 text-sm font-medium">{day.volume.toLocaleString()}</td>
                            <td className="px-4 py-2 text-sm">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                day.volume_ratio > 3 ? 'bg-red-100 text-red-800' :
                                day.volume_ratio > 2 ? 'bg-yellow-100 text-yellow-800' :
                                'bg-blue-100 text-blue-800'
                              }`}>
                                {day.volume_ratio.toFixed(2)}x
                              </span>
                            </td>
                            <td className="px-4 py-2 text-sm">${day.price.toFixed(2)}</td>
                            <td className="px-4 py-2 text-sm">
                              <span className={day.price_change_pct >= 0 ? 'text-green-600' : 'text-red-600'}>
                                {day.price_change_pct >= 0 ? '+' : ''}{day.price_change_pct.toFixed(2)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No unusual volume activity detected in the last 30 days
                  </div>
                )}
              </div>

              {/* Volume Ratio Chart */}
              <div className="p-6 bg-white rounded-lg border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Volume Ratio Over Time</h3>
                <div className="w-full h-[300px]">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={data.volume_analysis.unusual_days} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <XAxis 
                        dataKey="date" 
                        tick={{ fontSize: 12 }}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis tick={{ fontSize: 12 }} />
                      <Tooltip 
                        formatter={(value) => [`${Number(value).toFixed(2)}x`, 'Volume Ratio']}
                        labelFormatter={(label) => `Date: ${label}`}
                      />
                      <Line
                        type="monotone"
                        dataKey="volume_ratio"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === "darkpool" && (
            <div className="space-y-6">
              {/* Dark Pool Risk Assessment */}
              <div className="p-6 bg-white rounded-lg border shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Dark Pool Risk Assessment</h3>
                  <div className={`px-4 py-2 rounded-lg border ${getRiskLevelColor(data.dark_pool_analysis.risk_level)}`}>
                    <div className="font-bold">{data.dark_pool_analysis.risk_level.toUpperCase()} RISK</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{data.dark_pool_analysis.analysis_period_days}</div>
                    <div className="text-sm text-gray-600">Days Analyzed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{data.dark_pool_analysis.total_signals}</div>
                    <div className="text-sm text-gray-600">Total Signals</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {data.dark_pool_analysis.dark_pool_signals.length > 0 ? 
                        (data.dark_pool_analysis.total_signals / data.dark_pool_analysis.analysis_period_days * 7).toFixed(1) : '0'
                      }
                    </div>
                    <div className="text-sm text-gray-600">Signals/Week</div>
                  </div>
                </div>
              </div>

              {/* Dark Pool Signals Timeline */}
              <div className="p-6 bg-white rounded-lg border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Recent Dark Pool Signals</h3>
                
                {data.dark_pool_analysis.dark_pool_signals.length > 0 ? (
                  <div className="space-y-3">
                    {data.dark_pool_analysis.dark_pool_signals.map((signal, index) => (
                      <div key={index} className="p-4 border rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors">
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center space-x-3">
                            <div className={`w-3 h-3 rounded-full ${
                              signal.type === 'low_volume_price_movement' ? 'bg-red-500' : 'bg-yellow-500'
                            }`}></div>
                            <div>
                              <div className="font-medium text-sm">
                                {signal.type === 'low_volume_price_movement' ? 'Low Volume Price Movement' : 'Price Gap'}
                              </div>
                              <div className="text-xs text-gray-600">{signal.date}</div>
                            </div>
                          </div>
                          <div className="text-right text-sm">
                            {signal.price_change_pct && (
                              <div className={`font-medium ${
                                signal.price_change_pct >= 0 ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {signal.price_change_pct >= 0 ? '+' : ''}{signal.price_change_pct.toFixed(2)}%
                              </div>
                            )}
                            {signal.gap_pct && (
                              <div className="font-medium text-orange-600">
                                Gap: {signal.gap_pct.toFixed(2)}%
                              </div>
                            )}
                            <div className="text-gray-600 text-xs">
                              Vol: {signal.volume_ratio.toFixed(2)}x
                            </div>
                          </div>
                        </div>
                        <p className="text-sm text-gray-700">{signal.description}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No dark pool signals detected in the analysis period
                  </div>
                )}
              </div>

              {/* Signal Type Distribution */}
              {data.dark_pool_analysis.dark_pool_signals.length > 0 && (
                <div className="p-6 bg-white rounded-lg border shadow-sm">
                  <h3 className="text-lg font-semibold mb-4">Signal Type Distribution</h3>
                  <div className="w-full h-[250px]">
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart 
                        data={[
                          {
                            name: 'Low Volume Movement',
                            count: data.dark_pool_analysis.dark_pool_signals.filter(s => s.type === 'low_volume_price_movement').length,
                            color: '#dc2626'
                          },
                          {
                            name: 'Price Gaps',
                            count: data.dark_pool_analysis.dark_pool_signals.filter(s => s.type === 'price_gap').length,
                            color: '#d97706'
                          }
                        ]}
                        margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                      >
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip formatter={(value) => [`${value}`, 'Signal Count']} />
                        <Bar dataKey="count" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Refresh Button */}
          <div className="mt-8 text-center">
            <button
              onClick={() => refetch()}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              Refresh Analysis
            </button>
          </div>

          <div className="mt-6 text-sm text-gray-500 text-center">
            Analysis last updated: {new Date(data.timestamp).toLocaleString()}
          </div>
        </>
      )}
    </div>
  );
}
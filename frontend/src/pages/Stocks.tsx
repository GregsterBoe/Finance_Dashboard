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

interface PriceHistory {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface StockAssessment {
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  current_price: number;
  price_change: number;
  price_change_percent: number;
  market_cap: number;
  overall_score: number;
  tier: string;
  tier_info: {
    min_score: number;
    color: string;
    description: string;
  };
  key_metrics: {
    pe_ratio?: number;
    pb_ratio?: number;
    roe?: number;
    debt_to_equity?: number;
    dividend_yield?: number;
    beta?: number;
  };
  fundamentals: Record<string, any>;
  price_history: PriceHistory[];
  last_updated: string;
}

interface StockAssessmentResponse {
  timestamp: string;
  assessments: StockAssessment[];
  tiers: Record<string, StockAssessment[]>;
  summary: {
    total_stocks_assessed: number;
    average_score: number;
    tier_distribution: Record<string, number>;
    sector_analysis: Record<string, { count: number; avg_score: number }>;
    assessment_config: any;
  };
}

export default function Stocks() {
  const [expandedTiers, setExpandedTiers] = useState<Record<string, boolean>>({});

  const { data, isLoading, error } = useQuery<StockAssessmentResponse>({
    queryKey: ["stock-assessment"],
    queryFn: async () => {
      const res = await axios.get("http://127.0.0.1:8000/api/stock-assessment");
      return res.data;
    },
    retry: 1,
    retryDelay: 10000,
    staleTime: 60 * 1000,
    refetchInterval: 60 * 1000, // 1 min refresh
    refetchOnWindowFocus: false,
    refetchIntervalInBackground: false,
  });

  const toggleTier = (tier: string) => {
    setExpandedTiers((prev) => ({ ...prev, [tier]: !prev[tier] }));
  };

  if (isLoading) return <div className="p-6">Loading stock assessments...</div>;
  if (error) return <div className="p-6 text-red-500">Error loading stock assessments</div>;
  if (!data) return <div className="p-6">No stock assessment data available</div>;

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Stock Assessment Overview</h1>

      {/* Summary Section */}
      <div className="mb-8 p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border">
        <h2 className="text-xl font-semibold mb-4">Summary</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center mb-4">
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {data.summary.total_stocks_assessed}
            </div>
            <div className="text-sm text-gray-600">Total Stocks</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">
              {data.summary.average_score}
            </div>
            <div className="text-sm text-gray-600">Avg Score</div>
          </div>
          {Object.entries(data.summary.tier_distribution).map(([tier, count]) => (
            <div key={tier}>
              <div className="text-2xl font-bold">{count}</div>
              <div className="text-sm text-gray-600">{tier}</div>
            </div>
          ))}
        </div>

        {/* Sector Analysis */}
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">Sector Analysis</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
            {Object.entries(data.summary.sector_analysis).map(([sector, s]) => (
              <div
                key={sector}
                className="p-3 bg-white rounded-lg shadow-sm border text-center"
              >
                <div className="font-medium">{sector}</div>
                <div className="text-gray-600">{s.count} stocks</div>
                <div className="text-blue-600 font-bold">Avg {s.avg_score}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tier Panels */}
      <div className="space-y-6">
        {Object.entries(data.tiers).map(([tier, stocks]) => {
          const isExpanded = expandedTiers[tier] ?? true;
          const tierColor = data.summary.assessment_config.tiers[tier]?.color || "#6b7280";
          const tierDescription = data.summary.assessment_config.tiers[tier]?.description || "";

          return (
            <div key={tier} className="border rounded-lg shadow-sm bg-white">
              {/* Tier Header */}
              <div
                className="p-4 bg-gray-50 rounded-t-lg cursor-pointer hover:bg-gray-100 transition-colors"
                onClick={() => toggleTier(tier)}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: tierColor }}
                    />
                    <h2 className="text-xl font-semibold capitalize">{tier}</h2>
                    <span className="text-sm text-gray-500">
                      ({stocks.length} stocks)
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">{tierDescription}</div>
                </div>
              </div>

              {/* Tier Content */}
              {isExpanded && (
                <div className="p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {stocks.map((stock) => (
                      <div
                        key={stock.ticker}
                        className="p-4 border rounded-lg shadow-sm bg-white hover:shadow-md transition-shadow"
                      >
                        {/* Header */}
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <h3 className="font-semibold text-lg">{stock.name}</h3>
                            <p className="text-sm text-gray-500">{stock.ticker}</p>
                            <p className="text-xs text-gray-400">
                              {stock.sector} • {stock.industry}
                            </p>
                          </div>
                          <div className="text-right">
                            <div className="text-xl font-bold">
                              ${stock.current_price.toLocaleString()}
                            </div>
                            <div
                              className={`text-sm ${
                                stock.price_change >= 0
                                  ? "text-green-600"
                                  : "text-red-600"
                              }`}
                            >
                              {stock.price_change >= 0 ? "+" : ""}
                              {stock.price_change.toFixed(2)} (
                              {stock.price_change_percent.toFixed(2)}%)
                            </div>
                          </div>
                        </div>

                        {/* Chart */}
                        <div className="w-full h-[150px] mb-2">
                          <ResponsiveContainer width="100%" height={150}>
                            <LineChart data={stock.price_history}>
                              <XAxis dataKey="date" hide />
                              <YAxis hide domain={["dataMin", "dataMax"]} />
                              <Tooltip
                                formatter={(value: any) => [
                                  `$${Number(value).toFixed(2)}`,
                                  "Close",
                                ]}
                                labelFormatter={(label) => `Date: ${label}`}
                              />
                              <Line
                                type="monotone"
                                dataKey="close"
                                stroke={
                                  stock.price_change >= 0
                                    ? "#16a34a"
                                    : "#dc2626"
                                }
                                strokeWidth={2}
                                dot={false}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>

                        {/* Score Badge */}
                        <div className="flex items-center justify-between mb-2">
                          <span
                            className="px-3 py-1 rounded-full text-xs font-medium"
                            style={{
                              backgroundColor: `${tierColor}20`,
                              color: tierColor,
                            }}
                          >
                            Score: {stock.overall_score}
                          </span>
                          <span className="text-xs text-gray-500">
                            Updated:{" "}
                            {new Date(stock.last_updated).toLocaleTimeString()}
                          </span>
                        </div>

                        {/* Key Metrics */}
                        <div className="text-xs text-gray-600 grid grid-cols-2 gap-y-1">
                          <div>PE: {stock.key_metrics.pe_ratio ?? "—"}</div>
                          <div>PB: {stock.key_metrics.pb_ratio ?? "—"}</div>
                          <div>ROE: {stock.key_metrics.roe?.toFixed(1) ?? "—"}%</div>
                          <div>
                            D/E: {stock.key_metrics.debt_to_equity?.toFixed(2) ?? "—"}
                          </div>
                          <div>
                            Div: {stock.key_metrics.dividend_yield?.toFixed(2) ?? "—"}%
                          </div>
                          <div>Beta: {stock.key_metrics.beta ?? "—"}</div>
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

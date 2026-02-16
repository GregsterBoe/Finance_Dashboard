import { useState } from "react";
import { Treemap, ResponsiveContainer, Tooltip } from "recharts";

interface Transaction {
  booking_date: string;
  amount: number;
  category?: string;
  payee?: string;
  payer?: string;
  purpose?: string;
  transaction_type?: string;
}

interface Props {
  transactions: Transaction[];
}

interface CategoryData {
  name: string;
  value: number;
  count: number;
  color: string;
}

const CATEGORY_COLORS: Record<string, string> = {
  "Groceries": "#22c55e",
  "Dining & Restaurants": "#f97316",
  "Transportation": "#3b82f6",
  "Internet & Mobile": "#a855f7",
  "Rent & Housing": "#ef4444",
  "Insurance": "#6366f1",
  "Healthcare": "#ec4899",
  "Sport & Leisure": "#eab308",
  "Shopping": "#14b8a6",
  "Subscriptions": "#8b5cf6",
  "Income & Salary": "#10b981",
  "Transfer": "#64748b",
  "Investment": "#06b6d4",
  "Cash Withdrawal": "#78716c",
  "Miscellaneous": "#9ca3af",
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat("de-DE", {
    style: "currency",
    currency: "EUR",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
};

const formatDate = (dateStr: string): string => {
  if (!dateStr) return "N/A";
  try {
    const parts = dateStr.split(".");
    if (parts.length !== 3) return dateStr;
    let [day, month, year] = parts;
    if (year.length === 2) {
      const currentYear = new Date().getFullYear();
      const currentCentury = Math.floor(currentYear / 100) * 100;
      const twoDigitYear = parseInt(year);
      year =
        twoDigitYear > currentYear % 100
          ? String(currentCentury - 100 + twoDigitYear)
          : String(currentCentury + twoDigitYear);
    }
    return `${day}.${month}.${year}`;
  } catch {
    return dateStr;
  }
};

// Custom content renderer for treemap cells
const CustomTreemapContent = (props: any) => {
  const { x, y, width, height, name, value, color, onCategoryClick } = props;

  if (width < 4 || height < 4) return null;

  const showLabel = width > 60 && height > 30;
  const showValue = width > 80 && height > 50;

  return (
    <g
      onClick={() => onCategoryClick?.(name)}
      style={{ cursor: "pointer" }}
    >
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={color}
        stroke="#fff"
        strokeWidth={2}
        rx={4}
        ry={4}
      />
      {showLabel && (
        <text
          x={x + width / 2}
          y={y + height / 2 - (showValue ? 8 : 0)}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#fff"
          fontSize={width > 120 ? 14 : 11}
          fontWeight="bold"
          style={{ pointerEvents: "none" }}
        >
          {name}
        </text>
      )}
      {showValue && (
        <text
          x={x + width / 2}
          y={y + height / 2 + 14}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(255,255,255,0.85)"
          fontSize={width > 120 ? 13 : 10}
          style={{ pointerEvents: "none" }}
        >
          {formatCurrency(value)}
        </text>
      )}
    </g>
  );
};

// Custom tooltip
const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0].payload;

  return (
    <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
      <p className="text-sm font-bold text-gray-900">{data.name}</p>
      <p className="text-lg font-semibold text-red-600">
        {formatCurrency(data.value)}
      </p>
      <p className="text-xs text-gray-500">
        {data.count} transactions — click to view
      </p>
    </div>
  );
};

export default function CategoryBreakdown({ transactions }: Props) {
  const [viewMode, setViewMode] = useState<"expenses" | "income">("expenses");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const handleCategoryClick = (categoryName: string) => {
    setSelectedCategory(
      selectedCategory === categoryName ? null : categoryName
    );
  };

  // Aggregate transactions by category
  const aggregateByCategory = (type: "expenses" | "income"): CategoryData[] => {
    const categoryMap: Record<string, { total: number; count: number }> = {};

    transactions.forEach((txn) => {
      const category = txn.category || "Uncategorized";

      if (type === "expenses" && txn.amount >= 0) return;
      if (type === "income" && txn.amount <= 0) return;

      const amount = Math.abs(txn.amount);

      if (!categoryMap[category]) {
        categoryMap[category] = { total: 0, count: 0 };
      }
      categoryMap[category].total += amount;
      categoryMap[category].count += 1;
    });

    return Object.entries(categoryMap)
      .map(([name, data]) => ({
        name,
        value: Math.round(data.total * 100) / 100,
        count: data.count,
        color: CATEGORY_COLORS[name] || "#9ca3af",
      }))
      .sort((a, b) => b.value - a.value);
  };

  // Get transactions for a specific category
  const getTransactionsForCategory = (
    categoryName: string
  ): Transaction[] => {
    return transactions
      .filter((txn) => {
        const cat = txn.category || "Uncategorized";
        if (cat !== categoryName) return false;
        if (viewMode === "expenses" && txn.amount >= 0) return false;
        if (viewMode === "income" && txn.amount <= 0) return false;
        return true;
      })
      .sort((a, b) => Math.abs(b.amount) - Math.abs(a.amount));
  };

  const categoryData = aggregateByCategory(viewMode);
  const totalAmount = categoryData.reduce((sum, d) => sum + d.value, 0);

  const hasCategories = transactions.some((t) => t.category);

  // Reset selection when switching view mode
  const handleViewModeChange = (mode: "expenses" | "income") => {
    setViewMode(mode);
    setSelectedCategory(null);
  };

  if (!hasCategories) {
    return (
      <div className="bg-white rounded-lg p-6 border border-gray-200">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">
          Category Breakdown
        </h3>
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border border-gray-200">
          <div className="text-center">
            <p className="text-gray-500 mb-2">No categories available</p>
            <p className="text-sm text-gray-400">
              Use the "Categorize" button to categorize your transactions first
            </p>
          </div>
        </div>
      </div>
    );
  }

  const selectedTransactions = selectedCategory
    ? getTransactionsForCategory(selectedCategory)
    : [];

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-gray-900">
          Category Breakdown
        </h3>
        <div className="inline-flex rounded-lg border border-gray-300 bg-gray-50">
          <button
            onClick={() => handleViewModeChange("expenses")}
            className={`px-4 py-1.5 text-sm font-medium rounded-l-lg transition ${
              viewMode === "expenses"
                ? "bg-red-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            Expenses
          </button>
          <button
            onClick={() => handleViewModeChange("income")}
            className={`px-4 py-1.5 text-sm font-medium rounded-r-lg transition ${
              viewMode === "income"
                ? "bg-green-600 text-white"
                : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            Income
          </button>
        </div>
      </div>

      {/* Total */}
      <div className="mb-4 text-sm text-gray-600">
        Total {viewMode === "expenses" ? "Expenses" : "Income"}:{" "}
        <span
          className={`font-bold text-lg ${
            viewMode === "expenses" ? "text-red-600" : "text-green-600"
          }`}
        >
          {formatCurrency(totalAmount)}
        </span>
      </div>

      {/* Treemap */}
      {categoryData.length > 0 ? (
        <ResponsiveContainer width="100%" height={400}>
          <Treemap
            data={categoryData}
            dataKey="value"
            aspectRatio={4 / 3}
            content={
              <CustomTreemapContent
                onCategoryClick={handleCategoryClick}
              />
            }
            onClick={(data: any) => {
              if (data?.name) handleCategoryClick(data.name);
            }}
          >
            <Tooltip content={<CustomTooltip />} />
          </Treemap>
        </ResponsiveContainer>
      ) : (
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
          <p className="text-gray-500">
            No {viewMode === "expenses" ? "expenses" : "income"} found
          </p>
        </div>
      )}

      {/* Legend / Summary Table */}
      <div className="mt-6">
        <h4 className="font-semibold text-gray-900 mb-3">Details</h4>
        <div className="space-y-1">
          {categoryData.map((cat) => (
            <div key={cat.name}>
              <div
                onClick={() => handleCategoryClick(cat.name)}
                className={`flex items-center justify-between py-2 px-3 rounded-lg cursor-pointer transition ${
                  selectedCategory === cat.name
                    ? "bg-gray-100 ring-1 ring-gray-300"
                    : "hover:bg-gray-50"
                }`}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className="w-4 h-4 rounded"
                    style={{ backgroundColor: cat.color }}
                  />
                  <span className="text-sm font-medium text-gray-700">
                    {cat.name}
                  </span>
                  <span className="text-xs text-gray-400">
                    ({cat.count} txn)
                  </span>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-xs text-gray-400">
                    {((cat.value / totalAmount) * 100).toFixed(1)}%
                  </span>
                  <span className="text-sm font-semibold text-gray-900 w-24 text-right">
                    {formatCurrency(cat.value)}
                  </span>
                  <svg
                    className={`w-4 h-4 text-gray-400 transition-transform ${
                      selectedCategory === cat.name ? "rotate-180" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </div>
              </div>

              {/* Expanded transaction list */}
              {selectedCategory === cat.name && (
                <div className="ml-7 mt-1 mb-3 border-l-2 pl-4" style={{ borderColor: cat.color }}>
                  <div className="bg-gray-50 rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-xs text-gray-500 uppercase">
                          <th className="text-left py-2 px-3 font-medium">Date</th>
                          <th className="text-left py-2 px-3 font-medium">Payee</th>
                          <th className="text-left py-2 px-3 font-medium hidden md:table-cell">Purpose</th>
                          <th className="text-right py-2 px-3 font-medium">Amount</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {selectedTransactions.map((txn, idx) => (
                          <tr key={idx} className="hover:bg-white">
                            <td className="py-2 px-3 whitespace-nowrap text-gray-600">
                              {formatDate(txn.booking_date)}
                            </td>
                            <td className="py-2 px-3 text-gray-900">
                              <div className="max-w-xs truncate">
                                {txn.amount < 0
                                  ? txn.payee || "Unknown"
                                  : txn.payer || "Unknown"}
                              </div>
                            </td>
                            <td className="py-2 px-3 text-gray-500 hidden md:table-cell">
                              <div className="max-w-sm truncate">
                                {txn.purpose || "-"}
                              </div>
                            </td>
                            <td className="py-2 px-3 text-right whitespace-nowrap font-medium">
                              <span
                                className={
                                  txn.amount >= 0
                                    ? "text-green-600"
                                    : "text-red-600"
                                }
                              >
                                {formatCurrency(txn.amount)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

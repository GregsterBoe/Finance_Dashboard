interface UploadSummary {
  total_transactions: number;
  date_range: { from: string; to: string };
  balance_total: number;
  income_total: number;
  expense_total: number;
  transaction_types: Record<string, number>;
}

interface Props {
  summary: UploadSummary;
}

export default function SummaryStats({ summary }: Props) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("de-DE", {
      style: "currency",
      currency: "EUR",
    }).format(amount);
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr) return "N/A";
    try {
      // Check if it's already in dd.mm.yyyy or dd.mm.yy format
      if (dateStr.includes(".")) {
        const parts = dateStr.split(".");
        if (parts.length !== 3) return dateStr;

        let [day, month, year] = parts;

        // Handle 2-digit year
        if (year.length === 2) {
          const currentYear = new Date().getFullYear();
          const currentCentury = Math.floor(currentYear / 100) * 100;
          const twoDigitYear = parseInt(year);

          if (twoDigitYear > currentYear % 100) {
            year = String(currentCentury - 100 + twoDigitYear);
          } else {
            year = String(currentCentury + twoDigitYear);
          }
        }

        const date = new Date(`${year}-${month}-${day}`);
        if (isNaN(date.getTime())) return dateStr;
        return date.toLocaleDateString("de-DE");
      } else {
        // ISO format
        const date = new Date(dateStr);
        return date.toLocaleDateString("de-DE");
      }
    } catch {
      return dateStr;
    }
  };

  const transactionTypeEntries = Object.entries(summary.transaction_types)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-200">
          <div className="text-sm text-blue-600 font-medium mb-1">
            Total Transactions
          </div>
          <div className="text-3xl font-bold text-blue-900">
            {summary.total_transactions}
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 border border-green-200">
          <div className="text-sm text-green-600 font-medium mb-1">
            Total Income
          </div>
          <div className="text-3xl font-bold text-green-900">
            {formatCurrency(summary.income_total)}
          </div>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4 border border-red-200">
          <div className="text-sm text-red-600 font-medium mb-1">
            Total Expenses
          </div>
          <div className="text-3xl font-bold text-red-900">
            {formatCurrency(summary.expense_total)}
          </div>
        </div>

        <div
          className={`bg-gradient-to-br rounded-lg p-4 border ${
            summary.balance_total >= 0
              ? "from-emerald-50 to-emerald-100 border-emerald-200"
              : "from-orange-50 to-orange-100 border-orange-200"
          }`}
        >
          <div
            className={`text-sm font-medium mb-1 ${
              summary.balance_total >= 0
                ? "text-emerald-600"
                : "text-orange-600"
            }`}
          >
            Net Balance
          </div>
          <div
            className={`text-3xl font-bold ${
              summary.balance_total >= 0
                ? "text-emerald-900"
                : "text-orange-900"
            }`}
          >
            {formatCurrency(summary.balance_total)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <h4 className="font-semibold text-gray-900 mb-3">Date Range</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">From:</span>
              <span className="font-medium text-gray-900">
                {formatDate(summary.date_range.from)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">To:</span>
              <span className="font-medium text-gray-900">
                {formatDate(summary.date_range.to)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <h4 className="font-semibold text-gray-900 mb-3">
            Transaction Types
          </h4>
          <div className="space-y-2">
            {transactionTypeEntries.length > 0 ? (
              transactionTypeEntries.map(([type, count]) => (
                <div key={type} className="flex justify-between items-center">
                  <span className="text-gray-600 text-sm">
                    {type || "Unknown"}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="bg-blue-200 rounded-full h-2 w-20">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{
                          width: `${
                            (count / summary.total_transactions) * 100
                          }%`,
                        }}
                      ></div>
                    </div>
                    <span className="font-medium text-gray-900 text-sm w-8 text-right">
                      {count}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-sm">No transaction types</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

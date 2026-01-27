import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface Transaction {
  booking_date: string;
  amount: number;
}

interface Props {
  transactions: Transaction[];
}

interface ChartDataPoint {
  date: string;
  balance: number;
  dateObj: Date;
}

export default function BalanceChart({ transactions }: Props) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("de-DE", {
      style: "currency",
      currency: "EUR",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  };

  const parseDate = (dateStr: string): Date => {
    if (!dateStr) return new Date(0);

    try {
      const parts = dateStr.split(".");
      if (parts.length !== 3) return new Date(0);

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

      return new Date(`${year}-${month}-${day}`);
    } catch {
      return new Date(0);
    }
  };

  const formatDateForDisplay = (date: Date): string => {
    return date.toLocaleDateString("de-DE", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
    });
  };

  // Calculate cumulative balance over time
  const calculateBalanceOverTime = (): ChartDataPoint[] => {
    // Sort transactions by date (oldest first)
    const sortedTransactions = [...transactions].sort((a, b) => {
      const dateA = parseDate(a.booking_date);
      const dateB = parseDate(b.booking_date);
      return dateA.getTime() - dateB.getTime();
    });

    let cumulativeBalance = 0;
    const balanceData: ChartDataPoint[] = [];

    sortedTransactions.forEach((transaction) => {
      cumulativeBalance += transaction.amount;
      const dateObj = parseDate(transaction.booking_date);

      balanceData.push({
        date: formatDateForDisplay(dateObj),
        balance: cumulativeBalance,
        dateObj: dateObj,
      });
    });

    return balanceData;
  };

  const chartData = calculateBalanceOverTime();

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="text-sm font-medium text-gray-900">
            {payload[0].payload.date}
          </p>
          <p
            className={`text-lg font-bold ${
              payload[0].value >= 0 ? "text-green-600" : "text-red-600"
            }`}
          >
            {formatCurrency(payload[0].value)}
          </p>
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-gray-500">No transaction data available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200">
      <h3 className="text-xl font-semibold text-gray-900 mb-4">
        Balance Over Time
      </h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="date"
            stroke="#6b7280"
            style={{ fontSize: "12px" }}
            angle={-45}
            textAnchor="end"
            height={80}
            interval="preserveStartEnd"
          />
          <YAxis
            stroke="#6b7280"
            style={{ fontSize: "12px" }}
            tickFormatter={(value) => formatCurrency(value)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: "20px" }}
            iconType="line"
          />
          <Line
            type="monotone"
            dataKey="balance"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: "#3b82f6", r: 3 }}
            activeDot={{ r: 6 }}
            name="Balance"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-4 text-sm text-gray-600">
        <p>
          <strong>Starting Balance:</strong>{" "}
          {formatCurrency(chartData[0]?.balance || 0)}
        </p>
        <p>
          <strong>Ending Balance:</strong>{" "}
          {formatCurrency(chartData[chartData.length - 1]?.balance || 0)}
        </p>
        <p>
          <strong>Net Change:</strong>{" "}
          <span
            className={
              (chartData[chartData.length - 1]?.balance || 0) -
                (chartData[0]?.balance || 0) >=
              0
                ? "text-green-600 font-medium"
                : "text-red-600 font-medium"
            }
          >
            {formatCurrency(
              (chartData[chartData.length - 1]?.balance || 0) -
                (chartData[0]?.balance || 0)
            )}
          </span>
        </p>
      </div>
    </div>
  );
}

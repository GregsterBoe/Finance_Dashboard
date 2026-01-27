import { useState } from "react";

interface Transaction {
  booking_date: string;
  value_date: string;
  status: string;
  payer: string;
  payee: string;
  purpose: string;
  transaction_type: string;
  iban: string;
  amount: number;
  currency: string;
}

interface Pagination {
  offset: number;
  limit: number;
  total: number;
}

interface Props {
  transactions: Transaction[];
  pagination: Pagination;
  currentPage: number;
  onPageChange: (page: number) => void;
}

type SortField = "booking_date" | "amount" | "transaction_type";
type SortDirection = "asc" | "desc";

export default function TransactionTable({
  transactions,
  pagination,
  currentPage,
  onPageChange,
}: Props) {
  const [sortField, setSortField] = useState<SortField>("booking_date");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("de-DE", {
      style: "currency",
      currency: "EUR",
    }).format(amount);
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr) return "N/A";
    try {
      const parts = dateStr.split(".");
      if (parts.length !== 3) return dateStr;

      let [day, month, year] = parts;

      // Handle 2-digit year (convert to 4-digit)
      if (year.length === 2) {
        const currentYear = new Date().getFullYear();
        const currentCentury = Math.floor(currentYear / 100) * 100;
        const twoDigitYear = parseInt(year);

        // If year is greater than current year's last 2 digits, assume previous century
        if (twoDigitYear > currentYear % 100) {
          year = String(currentCentury - 100 + twoDigitYear);
        } else {
          year = String(currentCentury + twoDigitYear);
        }
      }

      const date = new Date(`${year}-${month}-${day}`);
      if (isNaN(date.getTime())) return dateStr;

      return date.toLocaleDateString("de-DE");
    } catch {
      return dateStr;
    }
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const sortedTransactions = [...transactions].sort((a, b) => {
    let aVal: any = a[sortField];
    let bVal: any = b[sortField];

    if (sortField === "booking_date") {
      try {
        const parseDate = (dateStr: string) => {
          const parts = dateStr.split(".");
          if (parts.length !== 3) return 0;

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

          return new Date(`${year}-${month}-${day}`).getTime();
        };

        aVal = parseDate(a.booking_date);
        bVal = parseDate(b.booking_date);
      } catch {
        aVal = a.booking_date;
        bVal = b.booking_date;
      }
    }

    if (aVal < bVal) return sortDirection === "asc" ? -1 : 1;
    if (aVal > bVal) return sortDirection === "asc" ? 1 : -1;
    return 0;
  });

  const totalPages = Math.ceil(pagination.total / pagination.limit);

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <span className="text-gray-400">↕</span>;
    }
    return (
      <span className="text-blue-600">
        {sortDirection === "asc" ? "↑" : "↓"}
      </span>
    );
  };

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort("booking_date")}
              >
                <div className="flex items-center space-x-1">
                  <span>Date</span>
                  <SortIcon field="booking_date" />
                </div>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Payer/Payee
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Purpose
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort("transaction_type")}
              >
                <div className="flex items-center space-x-1">
                  <span>Type</span>
                  <SortIcon field="transaction_type" />
                </div>
              </th>
              <th
                className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort("amount")}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Amount</span>
                  <SortIcon field="amount" />
                </div>
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedTransactions.map((transaction, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                  {formatDate(transaction.booking_date)}
                </td>
                <td className="px-4 py-3 text-sm text-gray-900">
                  <div className="max-w-xs truncate">
                    {transaction.amount < 0
                      ? transaction.payee || "Unknown"
                      : transaction.payer || "Unknown"}
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">
                  <div className="max-w-md truncate">
                    {transaction.purpose || "-"}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm">
                  <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                    {transaction.transaction_type || "Unknown"}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-right font-medium">
                  <span
                    className={
                      transaction.amount >= 0
                        ? "text-green-600"
                        : "text-red-600"
                    }
                  >
                    {transaction.amount >= 0 ? "+" : ""}
                    {formatCurrency(transaction.amount)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between border-t border-gray-200 pt-4">
          <div className="text-sm text-gray-600">
            Showing {pagination.offset + 1} to{" "}
            {Math.min(
              pagination.offset + pagination.limit,
              pagination.total
            )}{" "}
            of {pagination.total} transactions
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 0}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                currentPage === 0
                  ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              Previous
            </button>

            <div className="flex items-center space-x-1">
              {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 5) {
                  pageNum = i;
                } else if (currentPage < 3) {
                  pageNum = i;
                } else if (currentPage > totalPages - 3) {
                  pageNum = totalPages - 5 + i;
                } else {
                  pageNum = currentPage - 2 + i;
                }

                return (
                  <button
                    key={pageNum}
                    onClick={() => onPageChange(pageNum)}
                    className={`px-3 py-1 rounded-md text-sm font-medium ${
                      currentPage === pageNum
                        ? "bg-blue-600 text-white"
                        : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
                    }`}
                  >
                    {pageNum + 1}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages - 1}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                currentPage >= totalPages - 1
                  ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                  : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
              }`}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

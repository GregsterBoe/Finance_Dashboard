interface UploadSummary {
  total_transactions: number;
  date_range: { from: string; to: string };
  balance_total: number;
  income_total: number;
  expense_total: number;
  transaction_types: Record<string, number>;
}

interface UploadMetadata {
  upload_id: string;
  format: string;
  uploaded_at: string;
  filename: string;
  summary: UploadSummary;
}

interface Props {
  upload: UploadMetadata;
  onViewDetails: (uploadId: string) => void;
  onDelete: (uploadId: string) => void;
}

export default function UploadListCard({
  upload,
  onViewDetails,
  onDelete,
}: Props) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat("de-DE", {
      style: "currency",
      currency: "EUR",
    }).format(amount);
  };

  const formatDate = (dateStr: string) => {
    if (!dateStr) return "N/A";
    try {
      // Check if it's in dd.mm.yyyy or dd.mm.yy format
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

  const formatDateTime = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleString("de-DE");
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition p-5 border border-gray-200">
      <div className="flex justify-between items-start mb-4">
        <div className="flex-1">
          <h3 className="font-semibold text-lg text-gray-900 truncate">
            {upload.filename}
          </h3>
          <p className="text-xs text-gray-500 mt-1">
            {formatDateTime(upload.uploaded_at)}
          </p>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete(upload.upload_id);
          }}
          className="text-red-600 hover:text-red-700 ml-2"
          title="Delete upload"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
            />
          </svg>
        </button>
      </div>

      <div className="space-y-3 mb-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-blue-50 rounded p-2">
            <div className="text-xs text-blue-600 font-medium">
              Transactions
            </div>
            <div className="text-lg font-bold text-blue-900">
              {upload.summary.total_transactions}
            </div>
          </div>
          <div
            className={`rounded p-2 ${
              upload.summary.balance_total >= 0
                ? "bg-green-50"
                : "bg-red-50"
            }`}
          >
            <div
              className={`text-xs font-medium ${
                upload.summary.balance_total >= 0
                  ? "text-green-600"
                  : "text-red-600"
              }`}
            >
              Balance
            </div>
            <div
              className={`text-lg font-bold ${
                upload.summary.balance_total >= 0
                  ? "text-green-900"
                  : "text-red-900"
              }`}
            >
              {formatCurrency(upload.summary.balance_total)}
            </div>
          </div>
        </div>

        <div className="text-sm space-y-1">
          <div className="flex justify-between text-gray-600">
            <span>Income:</span>
            <span className="font-medium text-green-600">
              {formatCurrency(upload.summary.income_total)}
            </span>
          </div>
          <div className="flex justify-between text-gray-600">
            <span>Expenses:</span>
            <span className="font-medium text-red-600">
              {formatCurrency(upload.summary.expense_total)}
            </span>
          </div>
        </div>

        <div className="text-xs text-gray-600 pt-2 border-t border-gray-200">
          <div className="flex justify-between">
            <span>From:</span>
            <span className="font-medium">
              {formatDate(upload.summary.date_range.from)}
            </span>
          </div>
          <div className="flex justify-between mt-1">
            <span>To:</span>
            <span className="font-medium">
              {formatDate(upload.summary.date_range.to)}
            </span>
          </div>
        </div>
      </div>

      <button
        onClick={() => onViewDetails(upload.upload_id)}
        className="w-full py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition font-medium text-sm"
      >
        View Details
      </button>
    </div>
  );
}

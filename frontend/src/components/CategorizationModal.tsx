import { useState, useEffect } from "react";

interface Props {
  uploadId: string;
  filename: string;
  onClose: () => void;
  onDataRefresh: () => void;
}

interface ProgressUpdate {
  type: "progress" | "complete" | "error";
  current?: number;
  total?: number;
  status?: string;
  percentage?: number;
  total_categorized?: number;
  category_counts?: Record<string, number>;
  error?: string;
}

export default function CategorizationModal({
  uploadId,
  filename,
  onClose,
  onDataRefresh,
}: Props) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [currentCount, setCurrentCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    total_categorized: number;
    category_counts: Record<string, number>;
  } | null>(null);

  const startCategorization = async () => {
    setIsRunning(true);
    setError(null);
    setProgress(0);
    setStatus("Connecting to categorization service...");

    try {
      const API_BASE = "http://localhost:8000";
      const eventSource = new EventSource(
        `${API_BASE}/api/categorize/stream/${uploadId}?model=gemma3:4b`
      );

      eventSource.onmessage = (event) => {
        const data: ProgressUpdate = JSON.parse(event.data);

        if (data.type === "progress") {
          setProgress(data.percentage || 0);
          setStatus(data.status || "");
          setCurrentCount(data.current || 0);
          setTotalCount(data.total || 0);
        } else if (data.type === "complete") {
          setProgress(100);
          setStatus("Categorization complete!");
          setResult({
            total_categorized: data.total_categorized || 0,
            category_counts: data.category_counts || {},
          });
          setIsRunning(false);
          eventSource.close();

          // Refresh data in the background so table updates
          setTimeout(() => {
            onDataRefresh();
          }, 500);
        } else if (data.type === "error") {
          setError(data.error || "Unknown error occurred");
          setIsRunning(false);
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        setError("Connection to server lost");
        setIsRunning(false);
        eventSource.close();
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start categorization");
      setIsRunning(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-2xl w-full mx-4">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-semibold text-gray-900">
              Categorize Transactions
            </h2>
            <p className="text-sm text-gray-600 mt-1">{filename}</p>
          </div>
          {!isRunning && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          )}
        </div>

        <div className="space-y-6">
          {/* Start Button */}
          {!isRunning && !result && !error && (
            <div className="text-center py-8">
              <p className="text-gray-600 mb-6">
                This will use an AI model to automatically categorize all
                transactions in this upload.
              </p>
              <button
                onClick={startCategorization}
                className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition font-medium"
              >
                Start Categorization
              </button>
            </div>
          )}

          {/* Progress Bar */}
          {isRunning && (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>{status}</span>
                  <span>
                    {currentCount} / {totalCount}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                  <div
                    className="bg-blue-600 h-4 transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="text-center text-sm font-medium text-gray-700">
                  {progress.toFixed(1)}%
                </div>
              </div>

              <div className="flex items-center justify-center py-4">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start">
                <svg
                  className="w-5 h-5 text-red-600 mt-0.5 mr-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div>
                  <h4 className="font-medium text-red-900">Error</h4>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
              <button
                onClick={startCategorization}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition text-sm"
              >
                Retry
              </button>
            </div>
          )}

          {/* Success Result */}
          {result && (
            <div className="space-y-4">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-start">
                  <svg
                    className="w-5 h-5 text-green-600 mt-0.5 mr-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <div>
                    <h4 className="font-medium text-green-900">
                      Categorization Complete!
                    </h4>
                    <p className="text-sm text-green-700 mt-1">
                      Successfully categorized {result.total_categorized}{" "}
                      transactions
                    </p>
                  </div>
                </div>
              </div>

              {/* Category Breakdown */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-3">
                  Category Breakdown
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(result.category_counts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([category, count]) => (
                      <div
                        key={category}
                        className="flex justify-between items-center bg-white rounded p-2 text-sm"
                      >
                        <span className="text-gray-700">{category}</span>
                        <span className="font-medium text-gray-900">
                          {count}
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              <button
                onClick={onClose}
                className="w-full py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition font-medium"
              >
                Close
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

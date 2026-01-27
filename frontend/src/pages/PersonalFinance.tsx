import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import CSVUploadSection from "../components/CSVUploadSection";
import SummaryStats from "../components/SummaryStats";
import TransactionTable from "../components/TransactionTable";
import UploadListCard from "../components/UploadListCard";
import BalanceChart from "../components/BalanceChart";

const API_BASE = "http://localhost:8000";

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
  creditor_id?: string;
  mandate_reference?: string;
  customer_reference?: string;
}

type ViewMode = "list" | "upload" | "detail";
type DetailViewMode = "table" | "chart";

export default function PersonalFinance() {
  const [activeView, setActiveView] = useState<ViewMode>("list");
  const [detailViewMode, setDetailViewMode] = useState<DetailViewMode>("table");
  const [selectedUploadId, setSelectedUploadId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch all uploads
  const { data: uploadsData, isLoading: uploadsLoading } = useQuery({
    queryKey: ["uploads"],
    queryFn: async () => {
      const res = await axios.get(`${API_BASE}/api/csv/uploads`);
      return res.data;
    },
  });

  // Fetch specific upload details
  const { data: uploadDetail } = useQuery({
    queryKey: ["upload", selectedUploadId],
    queryFn: async () => {
      if (!selectedUploadId) return null;
      const res = await axios.get(
        `${API_BASE}/api/csv/uploads/${selectedUploadId}`
      );
      return res.data;
    },
    enabled: !!selectedUploadId && activeView === "detail",
  });

  // Fetch transactions for selected upload (paginated for table)
  const [currentPage, setCurrentPage] = useState(0);
  const pageSize = 50;

  const { data: transactionsData } = useQuery({
    queryKey: ["transactions", selectedUploadId, currentPage],
    queryFn: async () => {
      if (!selectedUploadId) return null;
      const res = await axios.get(
        `${API_BASE}/api/csv/uploads/${selectedUploadId}/transactions?offset=${
          currentPage * pageSize
        }&limit=${pageSize}`
      );
      return res.data;
    },
    enabled: !!selectedUploadId && activeView === "detail" && detailViewMode === "table",
  });

  // Fetch all transactions for chart (use max allowed limit of 1000)
  const { data: allTransactionsData } = useQuery({
    queryKey: ["all-transactions", selectedUploadId],
    queryFn: async () => {
      if (!selectedUploadId) return null;
      const res = await axios.get(
        `${API_BASE}/api/csv/uploads/${selectedUploadId}/transactions?offset=0&limit=1000`
      );
      return res.data;
    },
    enabled: !!selectedUploadId && activeView === "detail" && detailViewMode === "chart",
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(`${API_BASE}/api/csv/upload`, formData);
      return res.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["uploads"] });
      setSelectedUploadId(data.upload_id);
      setActiveView("detail");
      setCurrentPage(0);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (uploadId: string) => {
      await axios.delete(`${API_BASE}/api/csv/uploads/${uploadId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["uploads"] });
      setActiveView("list");
      setSelectedUploadId(null);
    },
  });

  const handleUpload = (file: File) => {
    uploadMutation.mutate(file);
  };

  const handleViewDetails = (uploadId: string) => {
    setSelectedUploadId(uploadId);
    setCurrentPage(0);
    setActiveView("detail");
  };

  const handleDelete = (uploadId: string) => {
    if (window.confirm("Are you sure you want to delete this upload?")) {
      deleteMutation.mutate(uploadId);
    }
  };

  const handleBackToList = () => {
    setActiveView("list");
    setSelectedUploadId(null);
    setCurrentPage(0);
  };

  const uploads: UploadMetadata[] = uploadsData?.uploads || [];

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6 flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Personal Finance</h1>
        {activeView === "list" && (
          <button
            onClick={() => setActiveView("upload")}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
          >
            Upload CSV
          </button>
        )}
        {activeView !== "list" && (
          <button
            onClick={handleBackToList}
            className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition"
          >
            ← Back to List
          </button>
        )}
      </div>

      {activeView === "upload" && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Upload Bank Export</h2>
          <CSVUploadSection
            onUpload={handleUpload}
            isUploading={uploadMutation.isPending}
            error={uploadMutation.error?.message}
          />
        </div>
      )}

      {activeView === "list" && (
        <div>
          {uploadsLoading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">Loading uploads...</p>
            </div>
          ) : uploads.length === 0 ? (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
              <div className="text-6xl mb-4">📊</div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                No Uploads Yet
              </h2>
              <p className="text-gray-600 mb-6">
                Upload your first bank export CSV to get started
              </p>
              <button
                onClick={() => setActiveView("upload")}
                className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition"
              >
                Upload CSV
              </button>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {uploads.map((upload) => (
                <UploadListCard
                  key={upload.upload_id}
                  upload={upload}
                  onViewDetails={handleViewDetails}
                  onDelete={handleDelete}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {activeView === "detail" && uploadDetail && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-gray-900">
                  {uploadDetail.filename}
                </h2>
                <p className="text-sm text-gray-600">
                  Uploaded on{" "}
                  {new Date(uploadDetail.uploaded_at).toLocaleString()}
                </p>
              </div>
              <button
                onClick={() => handleDelete(uploadDetail.upload_id)}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition"
              >
                Delete
              </button>
            </div>
            <SummaryStats summary={uploadDetail.summary} />
          </div>

          {/* View Toggle Buttons */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">View</h3>
              <div className="inline-flex rounded-lg border border-gray-300 bg-gray-50">
                <button
                  onClick={() => setDetailViewMode("table")}
                  className={`px-6 py-2 text-sm font-medium rounded-l-lg transition ${
                    detailViewMode === "table"
                      ? "bg-blue-600 text-white"
                      : "bg-white text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  📋 Table
                </button>
                <button
                  onClick={() => setDetailViewMode("chart")}
                  className={`px-6 py-2 text-sm font-medium rounded-r-lg transition ${
                    detailViewMode === "chart"
                      ? "bg-blue-600 text-white"
                      : "bg-white text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  📈 Balance Chart
                </button>
              </div>
            </div>
          </div>

          {/* Table View */}
          {detailViewMode === "table" && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-xl font-semibold mb-4">Transactions</h3>
              {transactionsData && (
                <TransactionTable
                  transactions={transactionsData.transactions}
                  pagination={transactionsData.pagination}
                  currentPage={currentPage}
                  onPageChange={setCurrentPage}
                />
              )}
            </div>
          )}

          {/* Chart View */}
          {detailViewMode === "chart" && allTransactionsData && (
            <BalanceChart transactions={allTransactionsData.transactions} />
          )}
        </div>
      )}
    </div>
  );
}

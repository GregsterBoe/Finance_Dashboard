import { useState, useRef } from "react";

interface Props {
  onUpload: (file: File) => void;
  isUploading: boolean;
  error?: string;
}

export default function CSVUploadSection({
  onUpload,
  isUploading,
  error,
}: Props) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.name.endsWith(".csv")) {
        setSelectedFile(file);
      } else {
        alert("Please select a CSV file");
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (file.name.endsWith(".csv")) {
        setSelectedFile(file);
      } else {
        alert("Please select a CSV file");
      }
    }
  };

  const handleUploadClick = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-4">
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition ${
          dragActive
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 bg-gray-50"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="hidden"
        />

        <div className="space-y-4">
          <div className="text-6xl">📄</div>
          <div>
            <p className="text-lg font-medium text-gray-900">
              {selectedFile
                ? `Selected: ${selectedFile.name}`
                : "Drag and drop your CSV file here"}
            </p>
            <p className="text-sm text-gray-600 mt-1">
              or click below to browse
            </p>
          </div>
          <button
            type="button"
            onClick={handleBrowseClick}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition"
            disabled={isUploading}
          >
            Browse Files
          </button>
        </div>
      </div>

      {selectedFile && (
        <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">✓</div>
            <div>
              <p className="font-medium text-gray-900">{selectedFile.name}</p>
              <p className="text-sm text-gray-600">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <button
            onClick={() => setSelectedFile(null)}
            className="text-red-600 hover:text-red-700"
            disabled={isUploading}
          >
            Remove
          </button>
        </div>
      )}

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">
            <span className="font-semibold">Error:</span> {error}
          </p>
        </div>
      )}

      <button
        onClick={handleUploadClick}
        disabled={!selectedFile || isUploading}
        className={`w-full py-3 rounded-md font-medium transition ${
          !selectedFile || isUploading
            ? "bg-gray-300 text-gray-500 cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-700"
        }`}
      >
        {isUploading ? (
          <span className="flex items-center justify-center">
            <svg
              className="animate-spin h-5 w-5 mr-3"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            Uploading...
          </span>
        ) : (
          "Upload and Parse"
        )}
      </button>

      <div className="text-sm text-gray-600 space-y-1">
        <p className="font-medium">Supported format:</p>
        <ul className="list-disc list-inside space-y-1 pl-2">
          <li>German bank export CSV files</li>
          <li>
            Must include columns: Buchungsdatum, Wertstellung, Status, etc.
          </li>
        </ul>
      </div>
    </div>
  );
}

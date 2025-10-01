import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface TrainingConfig {
  ticker: string;
  start_date: string;
  end_date: string;
  model_type: string;
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf: number;
}

interface TrainingResponse {
  status: string;
  model_id: string;
  ticker: string;
  training_metrics: {
    rmse: number;
    mae: number;
    r2_score: number;
    training_samples: number;
  };
  prediction: {
    date: string;
    predicted_close: number;
    last_close: number;
    predicted_change: number;
    predicted_change_pct: number;
  };
  feature_importance: Record<string, number>;
  training_period: {
    start: string;
    end: string;
    days: number;
  };
  timestamp: string;
}

export default function StockPricePredictor() {
  const [step, setStep] = useState(1);
  const [tickers, setTickers] = useState<string[]>([]);
  const [isLoadingTickers, setIsLoadingTickers] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<TrainingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [config, setConfig] = useState<TrainingConfig>({
    ticker: "",
    start_date: "",
    end_date: "",
    model_type: "decision_tree",
    max_depth: 5,
    min_samples_split: 2,
    min_samples_leaf: 1,
  });

  // Load available tickers
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/available-tickers")
      .then(res => res.json())
      .then(data => {
        setTickers(data.tickers);
        setIsLoadingTickers(false);
      })
      .catch(err => {
        console.error("Failed to load tickers:", err);
        setIsLoadingTickers(false);
      });
  }, []);

  const handleNext = () => {
    if (step === 1 && !config.ticker) {
      alert("Please select a stock");
      return;
    }
    if (step === 2 && (!config.start_date || !config.end_date)) {
      alert("Please select both start and end dates");
      return;
    }
    setStep(step + 1);
  };

  const handleBack = () => setStep(step - 1);

  const handleTrain = async () => {
    setIsTraining(true);
    setError(null);
    
    try {
      const response = await fetch("http://127.0.0.1:8000/api/train-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Training failed");
      }

      const data = await response.json();
      setTrainingResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during training");
    } finally {
      setIsTraining(false);
    }
  };

  const resetTraining = () => {
    setStep(1);
    setTrainingResult(null);
    setError(null);
  };

  const featureImportanceData = trainingResult
    ? Object.entries(trainingResult.feature_importance).map(([name, value]) => ({
        name,
        importance: value,
      }))
    : [];

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">ML Model Training</h1>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {[1, 2, 3, 4].map((s) => (
            <div key={s} className="flex items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step >= s
                    ? "bg-blue-600 text-white"
                    : "bg-gray-200 text-gray-600"
                }`}
              >
                {s}
              </div>
              {s < 4 && (
                <div
                  className={`w-24 h-1 ${
                    step > s ? "bg-blue-600" : "bg-gray-200"
                  }`}
                />
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 text-sm text-gray-600">
          <span>Select Stock</span>
          <span>Time Period</span>
          <span>Configure Model</span>
          <span>Train & Results</span>
        </div>
      </div>

      {/* Step 1: Select Stock */}
      {step === 1 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Step 1: Select Stock</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Choose a stock ticker
              </label>
              {isLoadingTickers ? (
                <div className="p-3 text-gray-500">Loading tickers...</div>
              ) : (
                <select
                    className="w-full p-3 border rounded-lg"
                    value={config.ticker}
                    onChange={(e) =>
                      setConfig({ ...config, ticker: e.target.value })
                    }
                    title="Choose a stock ticker"
                  >
                    <option value="">Select a ticker...</option>
                    {tickers.map((ticker) => (
                      <option key={ticker} value={ticker}>
                        {ticker}
                      </option>
                    ))}
                  </select>
              )}
            </div>
          </div>
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleNext}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Time Period */}
      {step === 2 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">
            Step 2: Select Training Period
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Start Date
              </label>
              <input
                type="date"
                className="w-full p-3 border rounded-lg"
                value={config.start_date}
                max={config.end_date || new Date().toISOString().split("T")[0]}
                onChange={(e) =>
                  setConfig({ ...config, start_date: e.target.value })
                }
                placeholder="Select start date"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                End Date
              </label>
              <input
                type="date"
                className="w-full p-3 border rounded-lg"
                value={config.end_date}
                min={config.start_date}
                max={new Date().toISOString().split("T")[0]}
                onChange={(e) =>
                  setConfig({ ...config, end_date: e.target.value })
                }
                placeholder="Select end date"
              />
            </div>
            <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded">
              <strong>Note:</strong> Training period must be at least 30 days.
              More data generally leads to better predictions.
            </div>
          </div>
          <div className="mt-6 flex justify-between">
            <button
              onClick={handleBack}
              className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
            >
              Back
            </button>
            <button
              onClick={handleNext}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Configure Model */}
      {step === 3 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">
            Step 3: Configure Model
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Model Type
              </label>
              <select
                className="w-full p-3 border rounded-lg bg-gray-100"
                value={config.model_type}
                disabled
                title="Currently only Decision Tree model is supported"
              >
                <option value="decision_tree">Decision Tree</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                More models coming soon
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Max Depth: {config.max_depth}
              </label>
              <input
                type="range"
                min="2"
                max="20"
                value={config.max_depth}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    max_depth: parseInt(e.target.value),
                  })
                }
                className="w-full"
                placeholder="Set max depth"
                title="Set the maximum depth for the decision tree"
              />
              <p className="text-xs text-gray-500">
                Maximum depth of the tree. Higher values may overfit.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Min Samples Split: {config.min_samples_split}
              </label>
              <input
                type="range"
                min="2"
                max="20"
                value={config.min_samples_split}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    min_samples_split: parseInt(e.target.value),
                  })
                }
                className="w-full"
                placeholder="Set range"
                title="Set the minimum samples required to split a node"
              />
              <p className="text-xs text-gray-500">
                Minimum samples required to split a node.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Min Samples Leaf: {config.min_samples_leaf}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={config.min_samples_leaf}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    min_samples_leaf: parseInt(e.target.value),
                  })
                }
                className="w-full"
                placeholder="Set range"
                title="Set the minimum samples required at a leaf node"
              />
              <p className="text-xs text-gray-500">
                Minimum samples required at a leaf node.
              </p>
            </div>
          </div>
          <div className="mt-6 flex justify-between">
            <button
              onClick={handleBack}
              className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
            >
              Back
            </button>
            <button
              onClick={handleNext}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 4: Train & Results */}
      {step === 4 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">
            Step 4: Train Model & View Results
          </h2>

          {!trainingResult && !isTraining && !error && (
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">Training Configuration</h3>
                <div className="text-sm space-y-1">
                  <p>
                    <strong>Ticker:</strong> {config.ticker}
                  </p>
                  <p>
                    <strong>Period:</strong> {config.start_date} to{" "}
                    {config.end_date}
                  </p>
                  <p>
                    <strong>Model:</strong> {config.model_type}
                  </p>
                  <p>
                    <strong>Max Depth:</strong> {config.max_depth}
                  </p>
                </div>
              </div>
              <div className="flex space-x-4">
                <button
                  onClick={handleBack}
                  className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                >
                  Back
                </button>
                <button
                  onClick={handleTrain}
                  className="flex-1 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold"
                >
                  Start Training
                </button>
              </div>
            </div>
          )}

          {isTraining && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-lg font-semibold">Training model...</p>
              <p className="text-sm text-gray-600 mt-2">
                This may take a few moments
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
              <p className="text-red-600 font-semibold">Training Failed</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
              <button
                onClick={() => {
                  setError(null);
                  setTrainingResult(null);
                }}
                className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                Try Again
              </button>
            </div>
          )}

          {trainingResult && (
            <div className="space-y-6">
              {/* Success Message */}
              <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                <p className="text-green-800 font-semibold">
                  ✓ Training Completed Successfully
                </p>
                <p className="text-sm text-green-700 mt-1">
                  Model ID: {trainingResult.model_id}
                </p>
              </div>

              {/* Prediction Card */}
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg border">
                <h3 className="text-lg font-semibold mb-4">
                  Next Day Prediction
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Prediction Date</p>
                    <p className="text-2xl font-bold">
                      {trainingResult.prediction.date}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Predicted Close</p>
                    <p className="text-2xl font-bold text-blue-600">
                      ${trainingResult.prediction.predicted_close}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Last Close</p>
                    <p className="text-xl">
                      ${trainingResult.prediction.last_close}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Predicted Change</p>
                    <p
                      className={`text-xl font-semibold ${
                        trainingResult.prediction.predicted_change >= 0
                          ? "text-green-600"
                          : "text-red-600"
                      }`}
                    >
                      {trainingResult.prediction.predicted_change >= 0
                        ? "+"
                        : ""}
                      ${trainingResult.prediction.predicted_change} (
                      {trainingResult.prediction.predicted_change_pct.toFixed(
                        2
                      )}
                      %)
                    </p>
                  </div>
                </div>
              </div>

              {/* Training Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white p-4 rounded-lg border">
                  <p className="text-sm text-gray-600">RMSE</p>
                  <p className="text-xl font-bold">
                    {trainingResult.training_metrics.rmse}
                  </p>
                </div>
                <div className="bg-white p-4 rounded-lg border">
                  <p className="text-sm text-gray-600">MAE</p>
                  <p className="text-xl font-bold">
                    {trainingResult.training_metrics.mae}
                  </p>
                </div>
                <div className="bg-white p-4 rounded-lg border">
                  <p className="text-sm text-gray-600">R² Score</p>
                  <p className="text-xl font-bold">
                    {trainingResult.training_metrics.r2_score}
                  </p>
                </div>
                <div className="bg-white p-4 rounded-lg border">
                  <p className="text-sm text-gray-600">Training Samples</p>
                  <p className="text-xl font-bold">
                    {trainingResult.training_metrics.training_samples}
                  </p>
                </div>
              </div>

              {/* Feature Importance */}
              <div className="bg-white p-6 rounded-lg border">
                <h3 className="text-lg font-semibold mb-4">
                  Top Feature Importance
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={featureImportanceData}>
                    <XAxis
                      dataKey="name"
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Training Period Info */}
              <div className="bg-gray-50 p-4 rounded-lg text-sm">
                <p>
                  <strong>Training Period:</strong>{" "}
                  {trainingResult.training_period.start} to{" "}
                  {trainingResult.training_period.end} (
                  {trainingResult.training_period.days} days)
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-4">
                <button
                  onClick={resetTraining}
                  className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold"
                >
                  Train New Model
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
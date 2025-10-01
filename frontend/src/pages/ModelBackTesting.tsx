import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface BacktestConfig {
  ticker: string;
  backtest_mode: string;
  backtest_days?: number;
  backtest_start_date?: string;
  backtest_end_date?: string;
  training_history_days: number;
  model_type: string;
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf: number;
  retrain_for_each_prediction: boolean;
}

interface BacktestResult {
  date: string;
  actual_close: number;
  predicted_close: number;
  error: number;
  error_pct: number;
  training_samples: number;
}

interface BacktestResponse {
  status: string;
  ticker: string;
  backtest_period: {
    start: string;
    end: string;
    total_days: number;
    training_history_days: number;
    retrain_for_each: boolean;
  };
  predictions: BacktestResult[];
  summary_metrics: {
    mae: number;
    rmse: number;
    mape: number;
    r2_score: number;
    directional_accuracy: number;
    total_predictions: number;
    avg_error: number;
    avg_error_pct: number;
  };
  model_config: {
    model_type: string;
    max_depth: number;
    min_samples_split: number;
    min_samples_leaf: number;
  };
  timestamp: string;
}

export default function ModelBacktesting() {
  const [step, setStep] = useState(1);
  const [tickers, setTickers] = useState<string[]>([]);
  const [isLoadingTickers, setIsLoadingTickers] = useState(true);
  const [isBacktesting, setIsBacktesting] = useState(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [config, setConfig] = useState<BacktestConfig>({
    ticker: "",
    backtest_mode: "standard",
    backtest_days: 30,
    training_history_days: 90,
    model_type: "decision_tree",
    max_depth: 5,
    min_samples_split: 2,
    min_samples_leaf: 1,
    retrain_for_each_prediction: false,
  });

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
    if (step === 2) {
      if (config.backtest_mode === "standard" && (!config.backtest_days || config.backtest_days < 1)) {
        alert("Please enter valid backtest days");
        return;
      }
      if (config.backtest_mode === "custom" && (!config.backtest_start_date || !config.backtest_end_date)) {
        alert("Please select both start and end dates");
        return;
      }
    }
    setStep(step + 1);
  };

  const handleBack = () => setStep(step - 1);

  const handleBacktest = async () => {
    setIsBacktesting(true);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/backtest-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Backtesting failed");
      }

      const data = await response.json();
      setBacktestResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during backtesting");
    } finally {
      setIsBacktesting(false);
    }
  };

  const resetBacktest = () => {
    setStep(1);
    setBacktestResult(null);
    setError(null);
  };

  const chartData = backtestResult
    ? backtestResult.predictions.map(p => ({
        date: p.date,
        actual: p.actual_close,
        predicted: p.predicted_close,
      }))
    : [];

  const errorChartData = backtestResult
    ? backtestResult.predictions.map(p => ({
        date: p.date,
        error_pct: p.error_pct,
      }))
    : [];

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Model Backtesting</h1>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          {[1, 2, 3, 4].map((s) => (
            <div key={s} className="flex items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                  step >= s ? "bg-purple-600 text-white" : "bg-gray-200 text-gray-600"
                }`}
              >
                {s}
              </div>
              {s < 4 && (
                <div className={`w-24 h-1 ${step > s ? "bg-purple-600" : "bg-gray-200"}`} />
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 text-sm text-gray-600">
          <span>Select Stock</span>
          <span>Backtest Period</span>
          <span>Configure Model</span>
          <span>Run & Results</span>
        </div>
      </div>

      {/* Step 1: Select Stock */}
      {step === 1 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Step 1: Select Stock</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Choose a stock ticker</label>
              {isLoadingTickers ? (
                <div className="p-3 text-gray-500">Loading tickers...</div>
              ) : (
                <select
                  className="w-full p-3 border rounded-lg"
                  value={config.ticker}
                  onChange={(e) => setConfig({ ...config, ticker: e.target.value })}
                  aria-label="Choose a stock ticker"
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
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Backtest Period */}
      {step === 2 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">
            Step 2: Configure Backtest Period
          </h2>
          <div className="space-y-4">
            {/* Mode Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Backtest Mode
              </label>
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="backtest_mode"
                    value="standard"
                    checked={config.backtest_mode === "standard"}
                    onChange={(e) =>
                      setConfig({ ...config, backtest_mode: e.target.value })
                    }
                    className="w-4 h-4"
                  />
                  <span>Standard (Last N days)</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="backtest_mode"
                    value="custom"
                    checked={config.backtest_mode === "custom"}
                    onChange={(e) =>
                      setConfig({ ...config, backtest_mode: e.target.value })
                    }
                    className="w-4 h-4"
                  />
                  <span>Custom Date Range</span>
                </label>
              </div>
            </div>

            {/* Standard Mode */}
            {config.backtest_mode === "standard" && (
              <div>
                <label className="block text-sm font-medium mb-2">
                  Number of Days to Backtest
                </label>
                <input
                  type="number"
                  min="1"
                  max="365"
                  className="w-full p-3 border rounded-lg"
                  value={config.backtest_days}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      backtest_days: parseInt(e.target.value),
                    })
                  }
                  placeholder="Enter number of days"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Test the model on the last N trading days
                </p>
              </div>
            )}

            {/* Custom Mode */}
            {config.backtest_mode === "custom" && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Backtest Start Date
                  </label>
                  <input
                    type="date"
                    className="w-full p-3 border rounded-lg"
                    value={config.backtest_start_date}
                    max={config.backtest_end_date || new Date().toISOString().split("T")[0]}
                    onChange={(e) =>
                      setConfig({ ...config, backtest_start_date: e.target.value })
                    }
                    placeholder="Select start date"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Backtest End Date
                  </label>
                  <input
                    type="date"
                    className="w-full p-3 border rounded-lg"
                    value={config.backtest_end_date}
                    min={config.backtest_start_date}
                    max={new Date().toISOString().split("T")[0]}
                    onChange={(e) =>
                      setConfig({ ...config, backtest_end_date: e.target.value })
                    }
                    placeholder="Select end date"
                  />
                </div>
              </>
            )}

            {/* Training History */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Training History: {config.training_history_days} days
              </label>
              <input
                type="range"
                min="30"
                max="365"
                value={config.training_history_days}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    training_history_days: parseInt(e.target.value),
                  })
                }
                className="w-full"
                title="Set training history days"
              />
              <p className="text-xs text-gray-500">
                Amount of historical data to use for training before each prediction
              </p>
            </div>

            {/* Retrain Option */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.retrain_for_each_prediction}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      retrain_for_each_prediction: e.target.checked,
                    })
                  }
                  className="w-5 h-5"
                />
                <div>
                  <span className="font-medium">Retrain model for each prediction</span>
                  <p className="text-xs text-gray-600 mt-1">
                    If unchecked, model is trained once and reused. If checked, model is
                    retrained for each day (slower but more realistic).
                  </p>
                </div>
              </label>
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
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
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
            Step 3: Configure Model Parameters
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
                <option value="decision_tree">Decision Tree Regressor</option>
              </select>
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
                title="Set max depth"
              />
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
                title="Set min samples split"
              />
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
                title="Set min samples leaf"
              />
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
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

            {/* Step 4: Run & Results */}
      {step === 4 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Step 4: Run Backtest & View Results</h2>

          {/* Run button */}
          {!backtestResult && !isBacktesting && (
            <div className="flex justify-center">
              <button
                onClick={handleBacktest}
                className="px-8 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Run Backtest
              </button>
            </div>
          )}

          {/* Loading */}
          {isBacktesting && (
            <div className="p-6 text-center text-gray-600">
              <div className="animate-spin h-8 w-8 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              Running backtest... please wait
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="p-4 mb-4 bg-red-100 text-red-700 rounded-lg">
              <p>{error}</p>
              <button
                onClick={resetBacktest}
                className="mt-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                Restart
              </button>
            </div>
          )}

          {/* Results */}
          {backtestResult && (
            <div className="space-y-8">
              {/* Metrics Summary */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg shadow-sm">
                  <p className="text-sm text-gray-600">MAE</p>
                  <p className="text-lg font-bold">{backtestResult.summary_metrics.mae.toFixed(4)}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg shadow-sm">
                  <p className="text-sm text-gray-600">RMSE</p>
                  <p className="text-lg font-bold">{backtestResult.summary_metrics.rmse.toFixed(4)}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg shadow-sm">
                  <p className="text-sm text-gray-600">MAPE</p>
                  <p className="text-lg font-bold">{(backtestResult.summary_metrics.mape * 100).toFixed(2)}%</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg shadow-sm">
                  <p className="text-sm text-gray-600">R² Score</p>
                  <p className="text-lg font-bold">{backtestResult.summary_metrics.r2_score.toFixed(4)}</p>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg shadow-sm">
                  <p className="text-sm text-gray-600">Directional Accuracy</p>
                  <p className="text-lg font-bold">{(backtestResult.summary_metrics.directional_accuracy * 100).toFixed(2)}%</p>
                </div>
              </div>

              {/* Chart: Actual vs Predicted */}
              <div>
                <h3 className="text-lg font-semibold mb-2">Actual vs Predicted Closing Prices</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <XAxis dataKey="date" hide />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="actual" stroke="#4F46E5" name="Actual" />
                      <Line type="monotone" dataKey="predicted" stroke="#10B981" name="Predicted" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart: Error % */}
              <div>
                <h3 className="text-lg font-semibold mb-2">Prediction Error (%)</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={errorChartData}>
                      <XAxis dataKey="date" hide />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="error_pct" stroke="#EF4444" name="Error %" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Reset button */}
              <div className="flex justify-center">
                <button
                  onClick={resetBacktest}
                  className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                >
                  Start New Backtest
                </button>
              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
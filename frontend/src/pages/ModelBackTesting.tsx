import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import ProgressIndicator from "../components/ProgressIndicator";
import MetricsDisplay from '../components/MetricsDisplay';
import type { TrainingMetrics } from '../types/metrics';

interface ModelConfig {
  model_type: string;
  max_depth: number;
  min_samples_split: number;
  min_samples_leaf: number;
  n_estimators: number;
  random_state: number;
  // LSTM parameters
  sequence_length?: number;
  hidden_size?: number;
  num_layers?: number;
  dropout?: number;
  learning_rate?: number;
  epochs?: number;
  batch_size?: number;
  validation_sequences?: number;
  early_stopping_patience?: number;
  use_validation?: boolean;
  // Directional loss parameters
  use_directional_loss?: boolean;
  directional_loss_type?: string;
  price_weight: number;
  direction_weight: number;
  direction_threshold?: number;

  // NEW: Focal loss specific parameters
  focal_alpha: number;
  focal_gamma: number;
  
  // NEW: Adaptive loss specific parameters
  initial_price_weight: number;
  target_direction_accuracy: number;
  adaptation_rate: number;
}

interface BacktestConfig {
  ticker: string;
  backtest_mode: string;
  backtest_days?: number;
  backtest_start_date?: string;
  backtest_end_date?: string;
  training_history_days: number;
  model_spec: ModelConfig;  // ← Changed from flat structure
  retrain_for_each_prediction: boolean;
  notes: string;
}

interface BacktestResult {
  date: string;
  actual_close: number;
  predicted_close: number;
  error: number;
  error_pct: number;
  training_samples: number;
  
  // NEW
  actual_return?: number;
  predicted_return?: number;
  return_error?: number;
  correct_direction?: boolean;
}

interface BacktestResponse {
  status: string;
  run_id: string;
  ticker: string;
  backtest_period: {
    start: string;
    end: string;
    total_days: number;
    training_history_days: number;
    retrain_for_each: boolean;
  };
  predictions: BacktestResult[];
  summary_metrics: TrainingMetrics;
  model_spec: ModelConfig;
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
  model_spec: {
    model_type: "decision_tree",
    max_depth: 5,
    min_samples_split: 2,
    min_samples_leaf: 1,
    n_estimators: 100,
    random_state: 42,
    // LSTM defaults
    sequence_length: 20,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.2,
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
    validation_sequences: 30,
    early_stopping_patience: 25,
    use_validation: true,
    // Directional loss defaults (ensure numeric values to avoid undefined)
    use_directional_loss: false,
    directional_loss_type: "standard",
    price_weight: 0.6,
    direction_weight: 0.4,
    direction_threshold: 0.0,
    
    // NEW: Focal loss defaults
    focal_alpha: 0.25,
    focal_gamma: 2.0,
    
    // NEW: Adaptive loss defaults
    initial_price_weight: 0.7,
    target_direction_accuracy: 60.0,
    adaptation_rate: 0.01,
  },
  retrain_for_each_prediction: false,
  notes: "",
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
    setBacktestResult(null);
  };

  const handleBacktestComplete = (result: BacktestResponse) => {
    setBacktestResult(result);
    setIsBacktesting(false);
  };

  const handleBacktestError = (errorMessage: string) => {
    setError(errorMessage);
    setIsBacktesting(false);
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

  // Helper function to update model config
  const handleModelConfigChange = (field: keyof ModelConfig, value: number | string | boolean) => {
    
    setConfig(prevConfig => ({
      ...prevConfig,
      model_spec: { 
        ...prevConfig.model_spec, 
        [field]: value 
      }
    }));
  };

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
                className="w-full p-3 border rounded-lg"
                value={config.model_spec.model_type}
                onChange={(e) => handleModelConfigChange('model_type', e.target.value)}
                title="Select model type"
              >
                <option value="decision_tree">Decision Tree Regressor</option>
                <option value="random_forest">Random Forest Regressor</option>
                <option value="linear_regression">Linear Regression</option>
                <option value="lstm">LSTM Neural Network</option>
              </select>
            </div>

            {config.model_spec.model_type === 'lstm' && (
              <>
              <div className="space-y-4 bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold">LSTM Parameters</h3>
                
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Sequence Length: {config.model_spec.sequence_length}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={config.model_spec.sequence_length}
                    onChange={(e) => handleModelConfigChange('sequence_length', parseInt(e.target.value))}
                    className="w-full"
                    title="Set sequence length"
                  />
                  <p className="text-xs text-gray-500">Past days to analyze</p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Hidden Size: {config.model_spec.hidden_size}
                  </label>
                  <input
                    type="range"
                    min="16"
                    max="256"
                    step="16"
                    value={config.model_spec.hidden_size}
                    onChange={(e) => handleModelConfigChange('hidden_size', parseInt(e.target.value))}
                    className="w-full"
                    title="Set hidden size"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Layers: {config.model_spec.num_layers}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={config.model_spec.num_layers}
                    onChange={(e) => handleModelConfigChange('num_layers', parseInt(e.target.value))}
                    className="w-full"
                    title="Set number of layers"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Epochs: {config.model_spec.epochs}
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="300"
                    step="10"
                    value={config.model_spec.epochs}
                    onChange={(e) => handleModelConfigChange('epochs', parseInt(e.target.value))}
                    className="w-full"
                    title="Set epochs"
                  />
                </div>

                <p className="text-xs text-yellow-700 bg-yellow-50 p-2 rounded">
                  Note: LSTM training is slower. Epochs are automatically reduced to 1/3 for backtesting.
                </p>
              </div>
              {/* NEW: Directional Loss Configuration */}
              <div className="border-t pt-4">
                <h4 className="font-medium text-sm mb-3">Directional Loss (Experimental)</h4>
                
                <div className="flex items-center mb-3">
                  <input
                    type="checkbox"
                    checked={config.model_spec.use_directional_loss}
                    onChange={(e) => handleModelConfigChange('use_directional_loss', e.target.checked)}
                    className="mr-2"
                    title="Enable directional loss"
                  />
                  <label className="text-sm">Enable directional loss for better direction accuracy</label>
                </div>
                
                {config.model_spec.use_directional_loss && (
                  <div className="space-y-3 ml-4 border-l-2 border-blue-200 pl-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Loss Type</label>
                      <select
                        value={config.model_spec.directional_loss_type}
                        onChange={(e) => handleModelConfigChange('directional_loss_type', e.target.value)}
                        className="w-full p-2 border rounded text-sm"
                        title="Select directional loss type"
                      >
                        <option value="standard">Standard (Balanced)</option>
                        <option value="focal">Focal (Focus on hard examples)</option>
                        <option value="adaptive">Adaptive (Auto-adjusting weights)</option>
                        <option value="ranking">Ranking (Relative performance)</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-1">
                        Price Weight: {config.model_spec.price_weight.toFixed(1)}
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.1"
                        value={config.model_spec.price_weight}
                        onChange={(e) => {
                          const priceWeight = parseFloat(e.target.value);
                          handleModelConfigChange('price_weight', priceWeight);
                          handleModelConfigChange('direction_weight', 1.0 - priceWeight);
                        }}
                        className="w-full"
                        title="Set price weight"
                      />
                      <p className="text-xs text-gray-500">
                        Higher = more focus on price accuracy vs direction accuracy
                      </p>
                    </div>
                    
                    <div className="text-xs text-yellow-700 bg-yellow-50 p-2 rounded">
                      <strong>Note:</strong> Directional loss is experimental. 
                      Start with standard type and 0.7 price weight.
                    </div>
                  </div>
                )}
              </div>
              </>
            )}

            {config.model_spec.model_type === 'random_forest' && (
              <div>
                <label className="block text-sm font-medium mb-2">
                  Number of Trees: {config.model_spec.n_estimators}
                </label>
                <input
                  type="range"
                  min="10"
                  max="500"
                  step="10"
                  value={config.model_spec.n_estimators}
                  onChange={(e) => handleModelConfigChange('n_estimators', parseInt(e.target.value))}
                  className="w-full"
                  title="Set number of trees"
                />
              </div>
            )}

            {(config.model_spec.model_type === 'decision_tree' || config.model_spec.model_type === 'random_forest') && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Max Depth: {config.model_spec.max_depth}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={config.model_spec.max_depth}
                    onChange={(e) =>
                      handleModelConfigChange('max_depth', parseInt(e.target.value))
                    }
                    className="w-full"
                    title="Set max depth"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Min Samples Split: {config.model_spec.min_samples_split}
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="20"
                    value={config.model_spec.min_samples_split}
                    onChange={(e) =>
                      handleModelConfigChange('min_samples_split', parseInt(e.target.value))
                    }
                    className="w-full"
                    title="Set min samples split"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Min Samples Leaf: {config.model_spec.min_samples_leaf}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={config.model_spec.min_samples_leaf}
                    onChange={(e) =>
                      handleModelConfigChange('min_samples_leaf', parseInt(e.target.value))
                    }
                    className="w-full"
                    title="Set min samples leaf"
                  />
                </div>
              </>
            )}

            <div>
              <label className="block text-sm font-medium mb-2">Notes (Optional)</label>
              <textarea
                className="w-full p-3 border rounded-lg"
                rows={3}
                value={config.notes}
                onChange={(e) => setConfig({ ...config, notes: e.target.value })}
                placeholder="Add notes about this backtest..."
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
          {!backtestResult && !isBacktesting && !error && (
            <div className="flex justify-center space-x-4">
              <button
                onClick={handleBack}
                className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Back
              </button>
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

          {/* Progress Indicator */}
          {isBacktesting && (
            <ProgressIndicator
              config={config}
              onComplete={handleBacktestComplete}
              onError={handleBacktestError}
            />
          )}

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded">
              <p className="text-red-800">Error: {error}</p>
              <button
                onClick={() => {
                  setError(null);
                  setIsBacktesting(false);
                }}
                className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Try Again
              </button>
            </div>
          )}

          {/* Results Display */}
          {backtestResult && (
            <div className="mt-6 space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Backtest Results</h3>
                <button
                  onClick={resetBacktest}
                  className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
                >
                  New Backtest
                </button>
              </div>

              {/* Metrics Summary */}
              <div>
                <div className="bg-white p-6 rounded-lg shadow-md">
                  <MetricsDisplay 
                    metrics={backtestResult.summary_metrics}
                    title="Summary Metrics"
                    includeDirectional={true}
                  />
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
                      <Line type="monotone" dataKey="actual" stroke="#4F46E5" name="Actual" strokeWidth={2} />
                      <Line type="monotone" dataKey="predicted" stroke="#10B981" name="Predicted" strokeWidth={2} strokeDasharray="5 5" />
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
                      <Line type="monotone" dataKey="error_pct" stroke="#EF4444" name="Error %" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Configuration Details */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-3">Backtest Configuration</h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-600">Period:</span>
                    <span className="ml-2 font-medium">
                      {backtestResult.backtest_period.start} to {backtestResult.backtest_period.end}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Total Days:</span>
                    <span className="ml-2 font-medium">{backtestResult.backtest_period.total_days}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Training History:</span>
                    <span className="ml-2 font-medium">{backtestResult.backtest_period.training_history_days} days</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Retrain Each:</span>
                    <span className="ml-2 font-medium">{backtestResult.backtest_period.retrain_for_each ? "Yes" : "No"}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Model:</span>
                    <span className="ml-2 font-medium">{backtestResult.model_spec.model_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Max Depth:</span>
                    <span className="ml-2 font-medium">{backtestResult.model_spec.max_depth}</span>
                  </div>
                </div>
              </div>

              {/* Reset button */}
              <div className="flex justify-center">
                <button
                  onClick={resetBacktest}
                  className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
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
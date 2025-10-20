// frontend/src/pages/StockPricePredictor.tsx
import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
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
  sequence_length: number;
  hidden_size: number;
  num_layers: number;
  dropout: number;
  learning_rate: number;
  epochs: number;
  batch_size: number;
  validation_sequences: number;  // NEW
  early_stopping_patience: number;  // NEW
  use_validation: boolean;  // NEW

  // NEW: Enhanced LSTM parameters
  bidirectional: boolean;
  use_layer_norm: boolean;
  use_residual: boolean;
  weight_decay: number;
  gradient_clip_norm: number;
  lr_patience: number;
  lr_factor: number;
  min_lr: number;
  
  // NEW: Directional loss parameters
  use_directional_loss: boolean;
  directional_loss_type: string;
  price_weight: number;
  direction_weight: number;
  direction_threshold: number;
  
  // NEW: Focal loss specific parameters
  focal_alpha: number;
  focal_gamma: number;
  
  // NEW: Adaptive loss specific parameters
  initial_price_weight: number;
  target_direction_accuracy: number;
  adaptation_rate: number;
}


interface TrainingConfig {
  ticker: string;
  start_date: string;
  end_date: string;
  model_spec: ModelConfig;
  notes: string;
}

interface TrainingResponse {
  status: string;
  run_id: string;
  ticker: string;
  training_metrics: TrainingMetrics;
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
  model_spec: ModelConfig;
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
  model_spec: {
    model_type: "decision_tree",
    max_depth: 5,
    min_samples_split: 2,
    min_samples_leaf: 1,
    n_estimators: 100,
    random_state: 42,
    
    // Basic LSTM defaults
    sequence_length: 30,
    hidden_size: 64,
    num_layers: 2,
    dropout: 0.2,
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
    validation_sequences: 30,
    early_stopping_patience: 10,
    use_validation: true,
    
    // NEW: Enhanced LSTM defaults
    bidirectional: true,
    use_layer_norm: true,
    use_residual: true,
    weight_decay: 0.00001,
    gradient_clip_norm: 1.0,
    lr_patience: 7,
    lr_factor: 0.5,
    min_lr: 0.000001,
    
    // NEW: Directional loss defaults
    use_directional_loss: false,
    directional_loss_type: 'standard',
    price_weight: 0.7,
    direction_weight: 0.3,
    direction_threshold: 0.0,
    
    // NEW: Focal loss defaults
    focal_alpha: 0.25,
    focal_gamma: 2.0,
    
    // NEW: Adaptive loss defaults
    initial_price_weight: 0.7,
    target_direction_accuracy: 60.0,
    adaptation_rate: 0.01,
  },
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

    console.log("Training config being sent:", JSON.stringify(config, null, 2)); // ADD THIS

    
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
          <h2 className="text-xl font-semibold mb-4">Step 3: Configure Model</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Model Type</label>
              <select
                className="w-full p-3 border rounded-lg"
                value={config.model_spec.model_type}
                onChange={(e) => handleModelConfigChange('model_type', e.target.value)}
                title="Select the machine learning model type"
              >
                <option value="decision_tree">Decision Tree Regressor</option>
                <option value="random_forest">Random Forest Regressor</option>
                <option value="linear_regression">Linear Regression</option>
                <option value="lstm">LSTM Neural Network</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {config.model_spec.model_type === 'decision_tree' && 'Single decision tree - fast, interpretable'}
                {config.model_spec.model_type === 'random_forest' && 'Ensemble of trees - more accurate, slower'}
                {config.model_spec.model_type === 'linear_regression' && 'Simple linear model - baseline'}
                {config.model_spec.model_type === 'lstm' && 'Deep learning - captures temporal patterns, requires more data'}
              </p>
            </div>

            {/* LSTM-specific parameters */}
            {config.model_spec.model_type === 'lstm' && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Sequence Length: {config.model_spec.sequence_length}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="5"
                    value={config.model_spec.sequence_length}
                    onChange={(e) => handleModelConfigChange('sequence_length', parseInt(e.target.value))}
                    className="w-full"
                    title="Number of past days to use for prediction (lookback window)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of past days to use for prediction (lookback window)
                  </p>
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
                    title="Number of LSTM hidden units (model capacity)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of LSTM hidden units (model capacity)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Number of Layers: {config.model_spec.num_layers}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={config.model_spec.num_layers}
                    onChange={(e) => handleModelConfigChange('num_layers', parseInt(e.target.value))}
                    className="w-full"
                    title="Number of stacked LSTM layers (model depth)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of stacked LSTM layers (model depth)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Epochs: {config.model_spec.epochs}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    step="10"
                    value={config.model_spec.epochs}
                    onChange={(e) => handleModelConfigChange('epochs', parseInt(e.target.value))}
                    className="w-full"
                    title="Number of training iterations (more = better fit but slower)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of training iterations (more = better fit but slower)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Learning Rate: {config.model_spec.learning_rate}
                  </label>
                  <input
                    type="range"
                    min="0.0001"
                    max="0.01"
                    step="0.0001"
                    value={config.model_spec.learning_rate}
                    onChange={(e) => handleModelConfigChange('learning_rate', parseFloat(e.target.value))}
                    className="w-full"
                    title="Step size for model updates (0.001 is typical)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Step size for model updates (0.001 is typical)
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Validation Sequences: {config.model_spec.validation_sequences}
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="5"
                    value={config.model_spec.validation_sequences}
                    onChange={(e) => handleModelConfigChange('validation_sequences', parseInt(e.target.value))}
                    className="w-full"
                    title="Number of sequences to use for validation (fixed, not percentage)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of sequences to use for validation (fixed, not percentage)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Early Stopping Patience: {config.model_spec.early_stopping_patience}
                  </label>
                  <input
                    type="range"
                    min="3"
                    max="30"
                    step="1"
                    value={config.model_spec.early_stopping_patience}
                    onChange={(e) => handleModelConfigChange('early_stopping_patience', parseInt(e.target.value))}
                    className="w-full"
                    title="Stop training if validation loss doesn't improve for N epochs"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Stop training if validation loss doesn't improve for N epochs
                  </p>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <label className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.model_spec.use_validation}
                      onChange={(e) => handleModelConfigChange('use_validation', e.target.checked)}
                      className="w-5 h-5"
                    />
                    <div>
                      <span className="font-medium">Use validation split</span>
                      <p className="text-xs text-gray-600 mt-1">
                        If unchecked, trains on all data (for production models). 
                        Recommended: keep checked for testing.
                      </p>
                    </div>
                  </label>
                </div>
                <div className="border-t pt-4 mt-6">
                  <h4 className="font-medium text-base mb-4 text-blue-800">Directional Loss (Experimental)</h4>
                  
                  <div className="bg-blue-50 p-4 rounded-lg mb-4">
                    <label className="flex items-start space-x-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.model_spec.use_directional_loss}
                        onChange={(e) => {
                          handleModelConfigChange('use_directional_loss', e.target.checked);
                          // Reset direction_weight when toggling
                          if (e.target.checked) {
                            handleModelConfigChange('direction_weight', 1.0 - config.model_spec.price_weight);
                          }
                        }}
                        className="w-5 h-5 mt-1"
                      />
                      <div>
                        <span className="font-medium">Enable directional loss</span>
                        <p className="text-xs text-gray-600 mt-1">
                          Optimizes for both price accuracy AND directional accuracy (up/down predictions).
                          May improve trading signal quality but could reduce absolute price accuracy.
                        </p>
                      </div>
                    </label>
                  </div>
                  
                  {config.model_spec.use_directional_loss && (
                    <div className="space-y-4 ml-4 border-l-2 border-blue-200 pl-4">
                      
                      {/* Loss Type Selection */}
                      <div>
                        <label className="block text-sm font-medium mb-2">Loss Type</label>
                        <select
                          value={config.model_spec.directional_loss_type}
                          onChange={(e) => handleModelConfigChange('directional_loss_type', e.target.value)}
                          className="w-full p-2 border rounded text-sm"
                          title="Select directional loss type"
                        >
                          <option value="standard">Standard (Balanced price + direction)</option>
                          <option value="focal">Focal (Focus on hard examples)</option>
                          <option value="adaptive">Adaptive (Auto-adjusting weights)</option>
                        </select>
                        <p className="text-xs text-gray-500 mt-1">
                          {config.model_spec.directional_loss_type === 'standard' && 'Combines MSE loss with direction prediction. Good starting point.'}
                          {config.model_spec.directional_loss_type === 'focal' && 'Focuses more on difficult-to-predict examples. May help with edge cases.'}
                          {config.model_spec.directional_loss_type === 'adaptive' && 'Automatically adjusts focus based on current directional accuracy.'}
                        </p>
                      </div>

                      {/* Standard Loss Parameters */}
                      {config.model_spec.directional_loss_type === 'standard' && (
                        <>
                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Price vs Direction Balance: {(config.model_spec.price_weight * 100).toFixed(0)}% price / {(config.model_spec.direction_weight * 100).toFixed(0)}% direction
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
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                              <span>More direction focus</span>
                              <span>More price focus</span>
                            </div>
                          </div>

                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Direction Threshold: {config.model_spec.direction_threshold}
                            </label>
                            <input
                              type="range"
                              min="-0.05"
                              max="0.05"
                              step="0.01"
                              value={config.model_spec.direction_threshold}
                              onChange={(e) => handleModelConfigChange('direction_threshold', parseFloat(e.target.value))}
                              className="w-full"
                              title="Set direction threshold"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              Threshold for direction calculation. 0 = any increase is "up", negative = require larger moves.
                            </p>
                          </div>
                        </>
                      )}

                      {/* Focal Loss Parameters */}
                      {config.model_spec.directional_loss_type === 'focal' && (
                        <>
                          <div>
                            <label className="block text-sm font-medium mb-2">
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
                          </div>

                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Focal Alpha: {config.model_spec.focal_alpha}
                            </label>
                            <input
                              type="range"
                              min="0.1"
                              max="0.9"
                              step="0.05"
                              value={config.model_spec.focal_alpha}
                              onChange={(e) => handleModelConfigChange('focal_alpha', parseFloat(e.target.value))}
                              className="w-full"
                              title="Set focal alpha"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              Weighting factor for rare examples (0.25 is typical)
                            </p>
                          </div>

                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Focal Gamma: {config.model_spec.focal_gamma}
                            </label>
                            <input
                              type="range"
                              min="0.5"
                              max="5.0"
                              step="0.5"
                              value={config.model_spec.focal_gamma}
                              onChange={(e) => handleModelConfigChange('focal_gamma', parseFloat(e.target.value))}
                              className="w-full"
                              title="Set focal gamma"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              Focus intensity on hard examples (2.0 is typical, higher = more focus)
                            </p>
                          </div>
                        </>
                      )}

                      {/* Adaptive Loss Parameters */}
                      {config.model_spec.directional_loss_type === 'adaptive' && (
                        <>
                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Initial Price Weight: {config.model_spec.initial_price_weight}
                            </label>
                            <input
                              type="range"
                              min="0.1"
                              max="0.9"
                              step="0.1"
                              value={config.model_spec.initial_price_weight}
                              onChange={(e) => handleModelConfigChange('initial_price_weight', parseFloat(e.target.value))}
                              className="w-full"
                              title="Initial weight for price loss"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              Starting balance - will adapt during training
                            </p>
                          </div>

                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Target Direction Accuracy: {config.model_spec.target_direction_accuracy}%
                            </label>
                            <input
                              type="range"
                              min="50"
                              max="80"
                              step="5"
                              value={config.model_spec.target_direction_accuracy}
                              onChange={(e) => handleModelConfigChange('target_direction_accuracy', parseFloat(e.target.value))}
                              className="w-full"
                              title="Set target direction accuracy"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              Target directional accuracy - loss will adapt to reach this
                            </p>
                          </div>

                          <div>
                            <label className="block text-sm font-medium mb-2">
                              Adaptation Rate: {config.model_spec.adaptation_rate}
                            </label>
                            <input
                              type="range"
                              min="0.001"
                              max="0.1"
                              step="0.001"
                              value={config.model_spec.adaptation_rate}
                              onChange={(e) => handleModelConfigChange('adaptation_rate', parseFloat(e.target.value))}
                              className="w-full"
                              title="Set adaptation rate for weights"
                            />
                            <p className="text-xs text-gray-500 mt-1">
                              How fast to adapt weights (0.01 is typical)
                            </p>
                          </div>
                        </>
                      )}

                      {/* Warning/Info Box */}
                      <div className="text-xs text-yellow-700 bg-yellow-50 p-3 rounded border border-yellow-200">
                        <strong>Experimental Feature:</strong> Directional loss is still being optimized. 
                        Start with "Standard" type and 60-70% price weight. Monitor both price accuracy and directional accuracy in results.
                        <br/><br/>
                        <strong>Expected impact:</strong> Better directional predictions but potentially slightly worse absolute price accuracy.
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

            {/* Traditional ML parameters - only show if not LSTM */}
            {config.model_spec.model_type !== 'lstm' && (
              <>
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
                      title="Number of trees in the forest"
                    />
                    <p className="text-xs text-gray-500">
                      More trees = better accuracy but slower training
                    </p>
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
                        onChange={(e) => handleModelConfigChange('max_depth', parseInt(e.target.value))}
                        className="w-full"
                        title="Maximum depth of the tree(s)"
                      />
                      <p className="text-xs text-gray-500">
                        Maximum depth of the tree. Higher values may overfit.
                      </p>
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
                        onChange={(e) => handleModelConfigChange('min_samples_split', parseInt(e.target.value))}
                        className="w-full"
                        title="Minimum samples required to split a node"
                      />
                      <p className="text-xs text-gray-500">
                        Minimum samples required to split an internal node.
                      </p>
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
                        onChange={(e) => handleModelConfigChange('min_samples_leaf', parseInt(e.target.value))}
                        className="w-full"
                        title="Minimum samples required at a leaf node"
                      />
                      <p className="text-xs text-gray-500">
                        Minimum samples required at a leaf node.
                      </p>
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
                    placeholder="Add notes about this training run..."
                  />
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-sm mb-2">Training Summary</h4>
                  <div className="text-xs space-y-1">
                    <p><strong>Stock:</strong> {config.ticker}</p>
                    <p><strong>Period:</strong> {config.start_date} to {config.end_date}</p>
                    <p><strong>Model:</strong> {config.model_spec.model_type}</p>
                    {config.model_spec.model_type === 'random_forest' && (
                      <p><strong>Trees:</strong> {config.model_spec.n_estimators}</p>
                    )}
                    {(config.model_spec.model_type === 'decision_tree' || config.model_spec.model_type === 'random_forest') && (
                      <>
                        <p><strong>Max Depth:</strong> {config.model_spec.max_depth}</p>
                        <p><strong>Min Split:</strong> {config.model_spec.min_samples_split}</p>
                        <p><strong>Min Leaf:</strong> {config.model_spec.min_samples_leaf}</p>
                      </>
                    )}
                  </div>
                </div>
              </>
            )}

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
        </div>
      )}

      {/* Step 4: Train & Results */}
      {step === 4 && !trainingResult && !error && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Step 4: Train Model</h2>
          <div className="text-center py-8">
            {!isTraining ? (
              <>
                <p className="text-gray-600 mb-6">
                  Ready to train your model with the selected configuration.
                </p>
                <div className="flex justify-center gap-4">
                  <button
                    onClick={handleBack}
                    className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
                  >
                    Back
                  </button>
                  <button
                    onClick={handleTrain}
                    className="px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold"
                  >
                    Start Training
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Training model... This may take a moment.</p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 p-6 rounded-lg">
          <h3 className="text-red-800 font-semibold mb-2">Training Error</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <div className="flex gap-4">
            <button
              onClick={handleBack}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
            >
              Go Back
            </button>
            <button
              onClick={resetTraining}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Start Over
            </button>
          </div>
        </div>
      )}

      {/* Training Results */}
      {trainingResult && (
        <div className="space-y-6">
          <div className="bg-green-50 border border-green-200 p-6 rounded-lg">
            <h3 className="text-green-800 font-semibold mb-2">✓ Training Complete</h3>
            <p className="text-green-700">
              Run ID: <span className="font-mono text-sm">{trainingResult.run_id}</span>
            </p>
            <p className="text-sm text-green-600 mt-1">
              Results have been saved to training_results.json
            </p>
          </div>

          {/* Prediction */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-semibold mb-4">Next Day Prediction</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Prediction Date</div>
                <div className="text-lg font-semibold">{trainingResult.prediction.date}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Last Close</div>
                <div className="text-lg font-semibold">${trainingResult.prediction.last_close}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Predicted Close</div>
                <div className="text-lg font-semibold">${trainingResult.prediction.predicted_close}</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-sm text-gray-600">Expected Change</div>
                <div className={`text-lg font-semibold ${trainingResult.prediction.predicted_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {trainingResult.prediction.predicted_change >= 0 ? '+' : ''}{trainingResult.prediction.predicted_change_pct.toFixed(2)}%
                </div>
              </div>
            </div>
          </div>

          {/* Training Metrics */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <MetricsDisplay 
              metrics={trainingResult.training_metrics}
              title="Training Metrics"
              includeDirectional={false}
            />
          </div>

          {/* Feature Importance */}
          {featureImportanceData.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-xl font-semibold mb-4">Top 10 Feature Importance</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportanceData}>
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-center gap-4">
            <button
              onClick={resetTraining}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Train Another Model
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
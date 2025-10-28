// frontend/src/components/MetricsDisplay.tsx
/**
 * Enhanced MetricsDisplay component for both price-based and return-based predictions
 * Maintains current styling while adding support for direction accuracy and return metrics
 */
import type { TrainingMetrics } from '../types/metrics';
import { METRIC_CONFIGS, getOrderedMetrics } from '../types/metrics';

interface MetricsDisplayProps {
  metrics: TrainingMetrics;
  title?: string;
  includeDirectional?: boolean;
  layout?: 'grid' | 'list';
  showDescriptions?: boolean;
}

export default function MetricsDisplay({
  metrics,
  title = "Training Metrics",
  includeDirectional = true,
  layout = 'grid',
  showDescriptions = false
}: MetricsDisplayProps) {
  const orderedMetrics = getOrderedMetrics(includeDirectional);
  
  // Filter out metrics that don't exist in the data
  const availableMetrics = orderedMetrics.filter(key => {
    const value = metrics[key];
    return value !== undefined && value !== null;
  });

  // Check if this is a return-based model
  const isReturnBased = metrics.metric_type === 'returns';

  if (layout === 'list') {
    return (
      <div className="space-y-3">
        {title && <h3 className="text-lg font-semibold mb-4">{title}</h3>}
        {availableMetrics.map(key => {
          const config = METRIC_CONFIGS[key];
          const value = metrics[key];
          
          return (
            <div key={key} className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <div>
                <span className="font-medium text-gray-700">{config.label}</span>
                {showDescriptions && (
                  <p className="text-xs text-gray-500 mt-1">{config.description}</p>
                )}
              </div>
              <span className="text-lg font-bold">
                {config.format(value as number)}
              </span>
            </div>
          );
        })}
        
        {/* Return-based indicator */}
        {isReturnBased && (
          <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
            <p className="text-xs text-blue-700">
              <span className="font-semibold">Note:</span> Metrics calculated on returns (not absolute prices)
            </p>
          </div>
        )}
      </div>
    );
  }

  // Grid layout (default) - matches your current style
  return (
    <div>
      {title && <h3 className="text-lg font-semibold mb-4">{title}</h3>}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {availableMetrics.map(key => {
          const config = METRIC_CONFIGS[key];
          const value = metrics[key];
          
          // Special styling for direction accuracy
          const isDirectionalAccuracy = key === 'direction_accuracy';
          const bgClass = isDirectionalAccuracy ? 'bg-green-50 border border-green-200' : config.colorClass;
          
          return (
            <div
              key={key}
              className={`p-4 rounded-lg shadow-sm ${bgClass}`}
              title={showDescriptions ? config.description : undefined}
            >
              <p className="text-sm text-gray-600 mb-1">{config.label}</p>
              <p className={`text-lg font-bold ${isDirectionalAccuracy ? 'text-green-700' : ''}`}>
                {config.format(value as number)}
              </p>
              {showDescriptions && (
                <p className="text-xs text-gray-500 mt-2">{config.description}</p>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Return-based indicator - only show for grid layout */}
      {isReturnBased && (
        <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-xs text-blue-700 flex items-center">
            <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <span>
              <span className="font-semibold">Return-based metrics:</span> RMSE/MAE calculated on returns (not absolute prices). 
              Focus on Direction Accuracy for trading insights.
            </span>
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Compact version for inline display
 */
export function MetricsInline({ metrics }: { metrics: TrainingMetrics }) {
  return (
    <div className="flex flex-wrap gap-4 text-sm">
      <span>
        <span className="text-gray-600">MAE:</span>{' '}
        <span className="font-semibold">{METRIC_CONFIGS.mae.format(metrics.mae)}</span>
      </span>
      <span>
        <span className="text-gray-600">RMSE:</span>{' '}
        <span className="font-semibold">{METRIC_CONFIGS.rmse.format(metrics.rmse)}</span>
      </span>
      <span>
        <span className="text-gray-600">R²:</span>{' '}
        <span className="font-semibold">{METRIC_CONFIGS.r2_score.format(metrics.r2_score)}</span>
      </span>
      {metrics.direction_accuracy !== undefined && (
        <span>
          <span className="text-gray-600">Dir. Acc:</span>{' '}
          <span className="font-semibold text-green-600">
            {METRIC_CONFIGS.direction_accuracy.format(metrics.direction_accuracy)}
          </span>
        </span>
      )}
      {metrics.metric_type === 'returns' && (
        <span className="text-xs text-blue-600 italic ml-2">
          (return-based)
        </span>
      )}
    </div>
  );
}
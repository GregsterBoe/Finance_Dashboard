// frontend/src/components/MetricsDisplay.tsx
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
  title = "Metrics",
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
      </div>
    );
  }

  // Grid layout (default)
  return (
    <div>
      {title && <h3 className="text-lg font-semibold mb-4">{title}</h3>}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {availableMetrics.map(key => {
          const config = METRIC_CONFIGS[key];
          const value = metrics[key];
          
          return (
            <div
              key={key}
              className={`p-4 rounded-lg shadow-sm ${config.colorClass}`}
              title={showDescriptions ? config.description : undefined}
            >
              <p className="text-sm text-gray-600 mb-1">{config.label}</p>
              <p className="text-lg font-bold">
                {config.format(value as number)}
              </p>
              {showDescriptions && (
                <p className="text-xs text-gray-500 mt-2">{config.description}</p>
              )}
            </div>
          );
        })}
      </div>
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
      {metrics.directional_accuracy !== undefined && (
        <span>
          <span className="text-gray-600">Dir. Acc:</span>{' '}
          <span className="font-semibold">
            {METRIC_CONFIGS.directional_accuracy.format(metrics.directional_accuracy)}
          </span>
        </span>
      )}
    </div>
  );
}
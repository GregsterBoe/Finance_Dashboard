// frontend/src/types/metrics.ts
/**
 * Centralized types and utilities for ML metrics.
 * Must match backend TrainingMetrics structure.
 */

export interface TrainingMetrics {
  rmse: number;
  mae: number;
  r2_score: number;
  mape: number;
  training_samples: number;
  directional_accuracy?: number;
}

export type MetricKey = keyof TrainingMetrics;

export interface MetricConfig {
  key: MetricKey;
  label: string;
  format: (value: number) => string;
  description: string;
  colorClass: string;
}

/**
 * Format a metric value with appropriate precision
 */
export function formatMetricValue(key: MetricKey, value: number | undefined): string {
  if (value === undefined || value === null) return 'N/A';
  
  switch (key) {
    case 'mape':
    case 'directional_accuracy':
      return `${value.toFixed(2)}%`;
    case 'rmse':
    case 'mae':
    case 'r2_score':
      return value.toFixed(4);
    case 'training_samples':
      return value.toLocaleString();
    default:
      return value.toString();
  }
}

/**
 * Get display configuration for all metrics
 */
export const METRIC_CONFIGS: Record<MetricKey, MetricConfig> = {
  rmse: {
    key: 'rmse',
    label: 'RMSE',
    format: (v) => formatMetricValue('rmse', v),
    description: 'Root Mean Squared Error - measures average prediction error magnitude',
    colorClass: 'bg-blue-50'
  },
  mae: {
    key: 'mae',
    label: 'MAE',
    format: (v) => formatMetricValue('mae', v),
    description: 'Mean Absolute Error - average absolute difference between predictions and actuals',
    colorClass: 'bg-blue-50'
  },
  r2_score: {
    key: 'r2_score',
    label: 'R² Score',
    format: (v) => formatMetricValue('r2_score', v),
    description: 'Coefficient of determination - proportion of variance explained by the model',
    colorClass: 'bg-blue-50'
  },
  mape: {
    key: 'mape',
    label: 'MAPE',
    format: (v) => formatMetricValue('mape', v),
    description: 'Mean Absolute Percentage Error - average percentage error',
    colorClass: 'bg-blue-50'
  },
  directional_accuracy: {
    key: 'directional_accuracy',
    label: 'Directional Accuracy',
    format: (v) => formatMetricValue('directional_accuracy', v),
    description: 'Percentage of correct up/down movement predictions',
    colorClass: 'bg-purple-50'
  },
  training_samples: {
    key: 'training_samples',
    label: 'Training Samples',
    format: (v) => formatMetricValue('training_samples', v),
    description: 'Number of samples used for training',
    colorClass: 'bg-gray-50'
  }
};

/**
 * Get metric interpretation (good/bad based on value)
 */
export function getMetricInterpretation(key: MetricKey, value: number): {
  status: 'excellent' | 'good' | 'fair' | 'poor';
  color: string;
} {
  switch (key) {
    case 'r2_score':
      if (value >= 0.9) return { status: 'excellent', color: 'text-green-600' };
      if (value >= 0.7) return { status: 'good', color: 'text-blue-600' };
      if (value >= 0.5) return { status: 'fair', color: 'text-yellow-600' };
      return { status: 'poor', color: 'text-red-600' };
    
    case 'directional_accuracy':
      if (value >= 70) return { status: 'excellent', color: 'text-green-600' };
      if (value >= 60) return { status: 'good', color: 'text-blue-600' };
      if (value >= 50) return { status: 'fair', color: 'text-yellow-600' };
      return { status: 'poor', color: 'text-red-600' };
    
    case 'mape':
      // Lower is better for error metrics
      if (value <= 5) return { status: 'excellent', color: 'text-green-600' };
      if (value <= 10) return { status: 'good', color: 'text-blue-600' };
      if (value <= 20) return { status: 'fair', color: 'text-yellow-600' };
      return { status: 'poor', color: 'text-red-600' };
    
    default:
      return { status: 'good', color: 'text-gray-600' };
  }
}

/**
 * Get ordered list of metrics for display
 */
export function getOrderedMetrics(includeDirectional: boolean = true): MetricKey[] {
  const base: MetricKey[] = ['mae', 'rmse', 'mape', 'r2_score'];
  if (includeDirectional) {
    base.push('directional_accuracy');
  }
  base.push('training_samples');
  return base;
}
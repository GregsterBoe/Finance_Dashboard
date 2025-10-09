from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

class ModelType(str, Enum):
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    LSTM = "lstm"  # Add this new type

class ModelConfig(BaseModel):
    """Shared model configuration for training and backtesting"""
    model_type: ModelType = Field(default=ModelType.DECISION_TREE)
    max_depth: Optional[int] = Field(default=5, ge=1, le=50)
    min_samples_split: Optional[int] = Field(default=2, ge=2, le=50)
    min_samples_leaf: Optional[int] = Field(default=1, ge=1, le=20)
    n_estimators: Optional[int] = Field(default=100, ge=10, le=500)
    random_state: int = Field(default=42)
    
    # LSTM-specific parameters
    sequence_length: Optional[int] = Field(default=20, ge=10, le=100)
    hidden_size: Optional[int] = Field(default=64, ge=16, le=256)
    num_layers: Optional[int] = Field(default=2, ge=1, le=5)
    dropout: Optional[float] = Field(default=0.2, ge=0.0, le=0.5)
    learning_rate: Optional[float] = Field(default=0.001, ge=0.0001, le=0.01)
    epochs: Optional[int] = Field(default=100, ge=10, le=500)
    batch_size: Optional[int] = Field(default=32, ge=8, le=128)
    validation_sequences: Optional[int] = Field(default=30, ge=10, le=100)
    early_stopping_patience: Optional[int] = Field(default=10, ge=3, le=30)
    use_validation: Optional[bool] = Field(default=True)
    
    class Config:
        use_enum_values = True

class TrainingMetrics(BaseModel):
    """Standard training metrics"""
    rmse: float
    mae: float
    r2_score: float
    training_samples: int
    mape: Optional[float] = None
    directional_accuracy: Optional[float] = None

class PredictionResult(BaseModel):
    """Single prediction result"""
    date: str
    predicted_close: float
    last_close: float
    predicted_change: float
    predicted_change_pct: float

class TrainingResult(BaseModel):
    """Complete training result with metadata"""
    run_id: str
    run_type: Literal["training", "backtest"]
    timestamp: str
    ticker: str
    model_spec: ModelConfig
    training_period: Dict[str, Any]
    metrics: TrainingMetrics
    prediction: Optional[PredictionResult] = None
    feature_importance: Optional[Dict[str, float]] = None
    backtest_predictions: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None

class ResultsManager:
    """Manages saving and loading training results"""
    
    def __init__(self, results_file: str = "training_results.json"):
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
    def save_result(self, result: TrainingResult) -> None:
        """Save a training result to the results file (excludes predictions)"""
        results = self.load_all_results()
        
        # Convert to dict and exclude prediction fields
        result_dict = result.dict(exclude={
            'prediction', 
            'feature_importance', 
            'backtest_predictions'
        })
        
        results.append(result_dict)
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all training results"""
        if not self.results_file.exists():
            return []
        
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def get_results_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """Get all results for a specific ticker"""
        all_results = self.load_all_results()
        return [r for r in all_results if r.get('ticker') == ticker]
    
    def get_results_by_type(self, run_type: str) -> List[Dict[str, Any]]:
        """Get all results of a specific type"""
        all_results = self.load_all_results()
        return [r for r in all_results if r.get('run_type') == run_type]
    
    def get_best_result(self, ticker: str, metric: str = "r2_score") -> Optional[Dict[str, Any]]:
        """Get the best result for a ticker based on a metric"""
        results = self.get_results_by_ticker(ticker)
        if not results:
            return None
        
        # Sort by metric (higher is better for r2_score, lower is better for mae/rmse)
        if metric in ["mae", "rmse", "mape"]:
            return min(results, key=lambda x: x.get('metrics', {}).get(metric, float('inf')))
        else:
            return max(results, key=lambda x: x.get('metrics', {}).get(metric, float('-inf')))
    
    def compare_runs(self, run_ids: List[str]) -> List[Dict[str, Any]]:
        """Compare multiple runs by their IDs"""
        all_results = self.load_all_results()
        return [r for r in all_results if r.get('run_id') in run_ids]
    
    def delete_result(self, run_id: str) -> bool:
        """Delete a specific result by run_id"""
        results = self.load_all_results()
        filtered_results = [r for r in results if r.get('run_id') != run_id]
        
        if len(filtered_results) == len(results):
            return False  # No result found
        
        with open(self.results_file, 'w') as f:
            json.dump(filtered_results, f, indent=2)
        return True

def generate_run_id(ticker: str, run_type: str) -> str:
    """Generate a unique run ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{ticker}_{run_type}_{timestamp}"
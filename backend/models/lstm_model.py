import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import os
import json
from datetime import datetime

class LSTMModel(nn.Module):
    """
    LSTM model for stock price prediction
    """
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out


class LSTMStockPredictor:
    """
    Complete LSTM-based stock price predictor with training and inference
    """
    def __init__(self, sequence_length: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, 
                 learning_rate: float = 0.001, model_dir: str = 'lstm_models'):
        """
        Args:
            sequence_length: Number of past days to use for prediction (default 20, reduced from 30)
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            model_dir: Directory to save model weights
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Device configuration (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.feature_scaler = MinMaxScaler()  # For features
        self.target_scaler = MinMaxScaler()   # For target prices
        self.feature_columns = []
        
        # Store validation data for metrics calculation
        self.X_val = None
        self.y_val = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features
        Same as your current implementation but organized for LSTM
        """
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        
        # Exponential moving averages
        df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        
        # Price momentum
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        
        # Volume features
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma_5'] + 1e-8)
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-8)
        df['close_open_ratio'] = df['Close'] / (df['Open'] + 1e-8)
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target: next day's close price (for prediction)
        df['target'] = df['Close'].shift(-1)
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training with proper scaling
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
        Returns:
            X: Input sequences (samples, sequence_length, features)
            y: Target values (samples, 1) - SCALED
        """
        df_clean = df[feature_cols + ['target']].dropna()
        
        # Get features and targets
        features = df_clean[feature_cols].values
        targets = df_clean['target'].values.reshape(-1, 1)  # Reshape for scaler
        
        # Scale BOTH features and targets
        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets)  # CRITICAL: Scale targets!
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])
            y.append(targets_scaled[i + self.sequence_length])  # Use SCALED target
        
        return np.array(X), np.array(y).flatten()  # Flatten y back to 1D
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              validation_sequences: int = 30, early_stopping_patience: int = 10,
              use_validation: bool = True) -> dict:
        """
        Train the LSTM model with optimized validation strategy
        Args:
            df: DataFrame with stock data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_sequences: Number of sequences to use for validation (fixed, not percentage)
            early_stopping_patience: Stop if validation doesn't improve for N epochs
            use_validation: If False, train on all data (for production)
        Returns:
            Training history dictionary
        """
        # Create features
        df_features = self.create_features(df)
        
        # Define feature columns (exclude OHLCV and target)
        self.feature_columns = [col for col in df_features.columns 
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                            'target', 'Dividends', 'Stock Splits']]
        
        # Prepare sequences (with scaling applied internally)
        X, y = self.prepare_sequences(df_features, self.feature_columns)
        
        print(f"Prepared {len(X)} sequences of length {self.sequence_length}")
        print(f"Feature dimension: {X.shape[2]}")
        
        # Split into train and validation
        if use_validation:
            # Check if we have enough data
            min_required = validation_sequences + 50  # Need at least 50 for training
            if len(X) < min_required:
                raise ValueError(
                    f"Need at least {min_required} sequences for training with validation. "
                    f"Got {len(X)}. Try reducing sequence_length or validation_sequences."
                )
            
            # Use last N sequences for validation (time-series safe)
            split_idx = len(X) - validation_sequences
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"Training sequences: {len(X_train)}")
            print(f"Validation sequences: {len(X_val)}")
        else:
            # Train on all data (production mode)
            X_train, X_val = X, None
            y_train, y_val = y, None
            print(f"Training on all {len(X_train)} sequences (no validation)")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        if use_validation:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            
            # Store for metrics calculation later
            self.X_val = X_val
            self.y_val = y_val
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if use_validation else None,
            'stopped_epoch': None
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Mini-batch training
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation and early stopping
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                self.model.train()
                
                history['val_loss'].append(val_loss.item())
                
                # Early stopping check
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Stop if patience exceeded
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    history['stopped_epoch'] = epoch + 1
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train: {avg_train_loss:.4f}, "
                          f"Val: {val_loss.item():.4f}, Patience: {patience_counter}/{early_stopping_patience}")
            else:
                # No validation - just print training loss
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # Restore best model if using validation
        if use_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with val_loss: {best_val_loss:.4f}")
        
        print("Training completed!")
        return history
    
    def get_validation_metrics(self) -> dict:
        """
        Calculate validation metrics using the stored validation data
        Returns dict with rmse, mae, r2, mape (all in actual price scale)
        """
        if self.X_val is None or self.y_val is None:
            raise ValueError("No validation data available. Train with use_validation=True first.")
        
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Get predictions on validation set (in SCALED space)
        self.model.eval()
        with torch.no_grad():
            val_predictions_scaled = self.model(self.X_val)
        
        # Convert to numpy (still SCALED)
        y_true_scaled = self.y_val.cpu().numpy().flatten()
        y_pred_scaled = val_predictions_scaled.cpu().numpy().flatten()
        
        # CRITICAL: Inverse transform to get ACTUAL PRICES
        y_true = self.target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics on ACTUAL PRICES
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # MAPE (with small epsilon to avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    def predict(self, df: pd.DataFrame, last_sequence_only: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model
        Args:
            df: DataFrame with stock data
            last_sequence_only: If True, predict only for the last sequence
        Returns:
            Predictions array (in ACTUAL PRICE scale, not scaled!)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create features
        df_features = self.create_features(df)
        df_clean = df_features[self.feature_columns].dropna()
        
        # Check if we have enough data
        if len(df_clean) < self.sequence_length:
            raise ValueError(
                f"Insufficient data after feature engineering. "
                f"Need at least {self.sequence_length} data points, "
                f"but only have {len(df_clean)} after removing NaN values. "
                f"Try using more historical data (at least {self.sequence_length + 30} days recommended)."
            )
        
        # Normalize features using fitted scaler
        features_scaled = self.feature_scaler.transform(df_clean.values)
        
        if last_sequence_only:
            # Predict only for the last sequence
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        else:
            # Create all possible sequences
            num_sequences = len(features_scaled) - self.sequence_length + 1
            
            if num_sequences <= 0:
                raise ValueError(
                    f"Cannot create sequences. Need at least {self.sequence_length} "
                    f"data points but have {len(features_scaled)}."
                )
            
            X = []
            for i in range(num_sequences):
                sequence = features_scaled[i:i + self.sequence_length]
                X.append(sequence)
            X = np.array(X)
        
        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor)
        
        # Convert to numpy (still in SCALED space [0, 1])
        predictions_scaled = predictions_scaled.cpu().numpy()
        
        # CRITICAL: Inverse transform to get ACTUAL PRICES
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def save_model(self, ticker: str, metadata: Optional[dict] = None):
        """
        Save model weights and configuration
        Args:
            ticker: Stock ticker symbol
            metadata: Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"lstm_{ticker}_{timestamp}"
        
        save_path = os.path.join(self.model_dir, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_path, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'feature_columns': self.feature_columns,
            'ticker': ticker,
            'timestamp': timestamp,
            'device': str(self.device)
        }
        
        if metadata:
            config.update(metadata)
        
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save BOTH scalers
        import joblib
        feature_scaler_path = os.path.join(save_path, "feature_scaler.pkl")
        joblib.dump(self.feature_scaler, feature_scaler_path)
        
        target_scaler_path = os.path.join(save_path, "target_scaler.pkl")
        joblib.dump(self.target_scaler, target_scaler_path)
        
        print(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, model_path: str):
        """
        Load model weights and configuration
        Args:
            model_path: Path to saved model directory
        """
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.sequence_length = config['sequence_length']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.feature_columns = config['feature_columns']
        
        # Load BOTH scalers
        import joblib
        feature_scaler_path = os.path.join(model_path, "feature_scaler.pkl")
        self.feature_scaler = joblib.load(feature_scaler_path)
        
        target_scaler_path = os.path.join(model_path, "target_scaler.pkl")
        self.target_scaler = joblib.load(target_scaler_path)
        
        # Initialize and load model
        input_size = len(self.feature_columns)
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        model_file = os.path.join(model_path, "model.pth")
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Download sample data
    ticker = "AAPL"
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    
    # Initialize predictor
    predictor = LSTMStockPredictor(
        sequence_length=20,  # Reduced for less data requirements
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        model_dir='lstm_models'
    )
    
    # Train model with early stopping
    history = predictor.train(
        df, 
        epochs=100, 
        batch_size=32,
        validation_sequences=30,  # Fixed number, not percentage
        early_stopping_patience=10,
        use_validation=True
    )
    
    # Make prediction for next day
    next_day_pred = predictor.predict(df, last_sequence_only=True)
    print(f"\nPredicted next day close price: ${next_day_pred[0][0]:.2f}")
    print(f"Last actual close price: ${df['Close'].iloc[-1]:.2f}")
    
    # Get validation metrics
    metrics = predictor.get_validation_metrics()
    print(f"\nValidation Metrics:")
    print(f"  RMSE: ${metrics['rmse']:.2f}")
    print(f"  MAE: ${metrics['mae']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Save model
    predictor.save_model(ticker, metadata={'training_samples': len(df)})
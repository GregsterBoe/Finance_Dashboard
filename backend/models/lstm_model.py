import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import os
import json
from datetime import datetime

from services.feature_service import get_feature_service, FeatureSet
from services.metrics_service import get_metrics_service
from models.directional_loss import create_directional_loss



class LSTMModel(nn.Module):
    """
    Enhanced LSTM model with bidirectional layers, layer normalization, and residual connections
    """
    def __init__(self, input_size: int,
                hidden_size: int = 64, 
                num_layers: int = 2,
                dropout: float = 0.2, 
                bidirectional: bool = True,
                use_layer_norm: bool = True,
                use_residual: bool = True):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Calculate LSTM output size (doubled if bidirectional)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # LSTM layers with bidirectional support
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Layer normalization for LSTM output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Residual connection projection (if input size != LSTM output size)
        if use_residual and input_size != lstm_output_size:
            self.residual_proj = nn.Linear(input_size, lstm_output_size)
        else:
            self.residual_proj = None
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with residual connection
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Layer normalization for FC layers
        if use_layer_norm:
            self.fc_layer_norm = nn.LayerNorm(hidden_size)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass with residual connections and layer normalization
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
    
        # --- FIX 2: Correctly combine final states ---
        if self.bidirectional:
            # h_n contains all layers' final states
            # We want the last layer's final forward state (index -2)
            # and the last layer's final backward state (index -1)
            last_layer_h_n_forward = h_n[-2, :, :]
            last_layer_h_n_backward = h_n[-1, :, :]
            # Concatenate them to get the full bidirectional representation
            lstm_out_combined = torch.cat((last_layer_h_n_forward, last_layer_h_n_backward), dim=1)
        else:
            # For a unidirectional LSTM, your original logic was fine.
            # We can get this from h_n as well.
            lstm_out_combined = h_n[-1, :, :]
            # Or, to be closer to your original:
            # lstm_out_combined = lstm_out[:, -1, :]
        
        # Apply layer normalization (using the new combined output)
        if self.use_layer_norm:
            # Use lstm_out_combined instead of lstm_out
            lstm_out_norm = self.layer_norm(lstm_out_combined)
        else:
            lstm_out_norm = lstm_out_combined

        # Residual connection
        if self.use_residual:
            residual = x[:, -1, :]
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            # Add residual to the normalized LSTM output
            lstm_out_final = lstm_out_norm + residual
        else:
            lstm_out_final = lstm_out_norm
        
        # --- FIX 3: (See below) Remove one dropout layer ---
        # First FC layer
        # out = self.dropout(lstm_out_final) # <-- Suggest removing this one
        out = self.fc1(lstm_out_final)
        out = self.relu(out)
        
        # Layer normalization for FC
        if self.use_layer_norm:
            out = self.fc_layer_norm(out)
        
        # Second FC layer (output)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class EarlyStopping:
    """Enhanced early stopping with multiple criteria"""
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True, monitor_train: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor_train = monitor_train
        
        self.best_loss = float('inf')
        self.best_train_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        
    def __call__(self, val_loss: float, train_loss: float, model: nn.Module):
        # Primary criterion: validation loss improvement
        improved = val_loss < (self.best_loss - self.min_delta)
        
        # Secondary criterion: avoid overfitting (train loss much better than val loss)
        if self.monitor_train and val_loss > train_loss * 2.0:
            # Potential overfitting detected
            improved = False
        
        if improved:
            self.best_loss = val_loss
            self.best_train_loss = train_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
        
        return self.should_stop


class LSTMStockPredictor:
    """
    Enhanced LSTM-based stock price predictor with advanced training features
    """
    def __init__(self, sequence_length: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, 
                 learning_rate: float = 0.001, model_dir: str = 'lstm_models',
                 bidirectional: bool = True, use_layer_norm: bool = True,
                 use_residual: bool = True, weight_decay: float = 1e-5,
                 gradient_clip_norm: float = 1.0, use_directional_loss: bool = False,
                 directional_loss_config: dict = None):
        """
        Args:
            sequence_length: Number of past days to use for prediction
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Initial learning rate
            model_dir: Directory to save model weights
            bidirectional: Use bidirectional LSTM
            use_layer_norm: Use layer normalization
            use_residual: Use residual connections
            weight_decay: L2 regularization strength
            gradient_clip_norm: Gradient clipping norm
            use_directional_loss: Whether to use directional loss instead of MSE
            directional_loss_config: Configuration for directional loss
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm

        self.use_directional_loss = use_directional_loss
        self.directional_loss_config = directional_loss_config or {
            'loss_type': 'standard',
            'price_weight': 0.7,
            'direction_weight': 0.3
        }
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store validation data for metrics
        self.X_val = None
        self.y_val = None
        
        print(f"Using device: {self.device}")
        print(f"Enhanced LSTM features: bidirectional={bidirectional}, "
              f"layer_norm={use_layer_norm}, residual={use_residual}")

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        Args:
            data: Feature data
            target: Target values
        Returns:
            X, y arrays for LSTM training
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
          validation_sequences: int = 50, early_stopping_patience: int = 25,
          use_validation: bool = True, lr_patience: int = 7, lr_factor: float = 0.5,
          min_lr: float = 1e-6) -> dict:
        """
        Enhanced training with clean directional loss implementation
        """
        print(f"Training enhanced LSTM model...")
        print(f"Dataset shape: {df.shape}")
        
        # Feature engineering
        feature_service = get_feature_service()
        df_features, expected_feature_cols = feature_service.prepare_for_lstm(
            df, feature_set=FeatureSet.LSTM
        )

        # Check what features are actually available after dropna
        df_clean = df_features[expected_feature_cols + ['Close', 'target']].dropna()
        available_features = [col for col in expected_feature_cols if col in df_clean.columns]

          # Prepare features and target
        self.feature_columns = available_features
        original_close_prices = df_clean['Close'].values # This is still needed for directional loss
        
        
        print(f"Expected features: {len(expected_feature_cols)}")
        print(f"Available features: {len(available_features)}")
        
        if len(df_clean) < self.sequence_length + 20:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length + 20} rows")
        
      

        # Assign features and the CORRECT target
        features = df_clean[self.feature_columns].values
        target = df_clean['target'].values
        
        # Scale data
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled)
        print(f"Created {len(X)} sequences")

        # Create mapping from sequence index to original data index
        sequence_to_original_idx = []
        for i in range(self.sequence_length, len(features_scaled)):
            sequence_to_original_idx.append(i - 1)  # Index of the "current" day (last day in sequence)
        
        # Train/validation split
        if use_validation:
            if len(X) < validation_sequences + 20:
                raise ValueError(f"Not enough sequences for validation")
            
            split_idx = len(X) - validation_sequences
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Split the sequence index mapping too
            train_seq_indices = sequence_to_original_idx[:split_idx]
            val_seq_indices = sequence_to_original_idx[split_idx:]
            
            print(f"Training sequences: {len(X_train)}")
            print(f"Validation sequences: {len(X_val)}")
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None
            train_seq_indices = sequence_to_original_idx
            val_seq_indices = None
            print(f"Training on all {len(X_train)} sequences (no validation)")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        if use_validation:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
            self.X_val = X_val
            self.y_val = y_val
        
        # Initialize model, optimizer, scheduler, etc.
        input_size = X.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_layer_norm=self.use_layer_norm,
            use_residual=self.use_residual
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr
        )
        
        # Loss function
        if self.use_directional_loss:
            loss_config = self._filter_loss_config(self.directional_loss_config)
            criterion = create_directional_loss(**loss_config)
            print(f"Using directional loss: {self.directional_loss_config['loss_type']}")
            print(f"Filtered config: {loss_config}")
        else:
            criterion = nn.MSELoss()
            print("Using standard MSE loss")
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-6,
            restore_best_weights=True,
            monitor_train=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'train_direction_accuracy': [],
            'val_direction_accuracy': [],
            'price_loss': [],
            'direction_loss': []
        }
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_losses = []
            epoch_direction_acc = []
            epoch_price_loss = []
            epoch_direction_loss = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Handle directional loss
                if self.use_directional_loss:
                    batch_prev_close_list = []
                    for j in range(len(batch_X)):
                        seq_idx = i + j
                        if seq_idx < len(train_seq_indices):
                            original_idx = train_seq_indices[seq_idx]
                            original_close = original_close_prices[original_idx]
                            # Scale to match target scaling
                            scaled_close = self.target_scaler.transform([[original_close]])[0][0]
                            batch_prev_close_list.append(scaled_close)
                        else:
                            # Should not happen, but fallback
                            batch_prev_close_list.append(batch_y[j].item())
                    
                    batch_prev_close = torch.tensor(batch_prev_close_list, device=self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.use_directional_loss:
                    loss, loss_components = criterion(outputs, batch_y, batch_prev_close)
                    epoch_price_loss.append(loss_components['price_loss'])
                    epoch_direction_loss.append(loss_components['direction_loss'])
                    epoch_direction_acc.append(loss_components['direction_accuracy'])
                else:
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Calculate epoch metrics
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            if self.use_directional_loss:
                history['train_direction_accuracy'].append(np.mean(epoch_direction_acc))
                history['price_loss'].append(np.mean(epoch_price_loss))
                history['direction_loss'].append(np.mean(epoch_direction_loss))
            
            # Validation phase
            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    
                    if self.use_directional_loss:
                        # Apply same approach for validation
                        val_prev_close_list = []
                        for j in range(len(X_val)):
                            if j < len(val_seq_indices):
                                original_idx = val_seq_indices[j]
                                original_close = original_close_prices[original_idx]
                                scaled_close = self.target_scaler.transform([[original_close]])[0][0]
                                val_prev_close_list.append(scaled_close)
                            else:
                                # Fallback
                                val_prev_close_list.append(y_val[j].item())
                        
                        val_prev_close = torch.tensor(val_prev_close_list, device=self.device).unsqueeze(1)
                        val_loss, val_loss_components = criterion(val_outputs, y_val, val_prev_close)
                        history['val_direction_accuracy'].append(val_loss_components['direction_accuracy'])
                    else:
                        val_loss = criterion(val_outputs, y_val)
                
                history['val_loss'].append(val_loss.item())
                scheduler.step(val_loss)
                
                # Early stopping check
                if early_stopping(val_loss.item(), avg_train_loss, self.model):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                # Progress reporting
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f} - "
                        f"Val Loss: {val_loss.item():.6f} - "
                        f"LR: {current_lr:.8f}")
                    
                    if self.use_directional_loss and len(epoch_direction_acc) > 0:
                        print(f"  Train Dir Acc: {np.mean(epoch_direction_acc):.2f}% - "
                            f"Val Dir Acc: {val_loss_components['direction_accuracy']:.2f}%")
            else:
                # No validation
                scheduler.step(avg_train_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f} - "
                        f"LR: {current_lr:.8f}")
            
            # Track learning rate
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print("Training completed!")
        return history
    
    def get_validation_metrics(self) -> dict:
        """Calculate validation metrics for return-based predictions"""
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
        
        # CRITICAL: Inverse transform to get ACTUAL RETURNS (not prices!)
        y_true_returns = self.target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred_returns = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics on returns
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse_returns = np.sqrt(mean_squared_error(y_true_returns, y_pred_returns))
        mae_returns = mean_absolute_error(y_true_returns, y_pred_returns)
        r2_returns = r2_score(y_true_returns, y_pred_returns)
        
        # Calculate directional accuracy (did we predict the right direction?)
        correct_direction = np.sum((y_true_returns > 0) == (y_pred_returns > 0))
        direction_accuracy = (correct_direction / len(y_true_returns)) * 100
        
        # MAPE on returns (absolute percentage error)
        # Be careful with MAPE when returns can be near zero
        non_zero_mask = np.abs(y_true_returns) > 1e-6
        if np.any(non_zero_mask):
            mape_returns = np.mean(np.abs((y_true_returns[non_zero_mask] - y_pred_returns[non_zero_mask]) / 
                                        y_true_returns[non_zero_mask])) * 100
        else:
            mape_returns = 0.0
        
        return {
            'rmse': rmse_returns,
            'mae': mae_returns,
            'r2_score': r2_returns,
            'mape': mape_returns,
            'direction_accuracy': direction_accuracy,
            'metric_type': 'returns'  # Flag to indicate these are return-based metrics
        }
    
    def predict(self, df: pd.DataFrame, last_sequence_only: bool = True) -> np.ndarray:
        """
        Original predict method - now returns PRICES for backward compatibility.
        Internally predicts returns and converts to prices.
        """
        result = self.predict_with_price_conversion(df, last_sequence_only)
        
        if last_sequence_only:
            return np.array([[result['predicted_price']]])
        else:
            return result['predicted_prices'].reshape(-1, 1)
        
    def predict_with_price_conversion(self, df: pd.DataFrame, last_sequence_only: bool = True) -> dict:
        """
        Make predictions and convert returns to actual prices.
        
        Returns:
            dict with:
                - predicted_returns: The raw return predictions (log returns)
                - predicted_prices: Converted to actual price predictions
                - last_close: The last known close price used for conversion
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create features
        feature_service = get_feature_service()
        df_features, _ = feature_service.prepare_for_lstm(
            df,
            feature_set=FeatureSet.LSTM  # Changed from ADVANCED
        )
        
        df_clean = df_features[self.feature_columns + ['Close']].dropna()
        
        # Check if we have enough data
        if len(df_clean) < self.sequence_length:
            raise ValueError(
                f"Insufficient data after feature engineering. "
                f"Need at least {self.sequence_length} data points, "
                f"but only have {len(df_clean)} after removing NaN values."
            )
        
        # Get the last close price for conversion
        last_close = df_clean['Close'].iloc[-1]
        
        # Normalize features using fitted scaler
        features_scaled = self.feature_scaler.transform(df_clean[self.feature_columns].values)
        
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
            close_prices = []
            for i in range(num_sequences):
                sequence = features_scaled[i:i + self.sequence_length]
                X.append(sequence)
                # Store the close price at the end of each sequence for conversion
                close_prices.append(df_clean['Close'].iloc[i + self.sequence_length - 1])
                
            X = np.array(X)
        
        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor)
        
        # Convert to numpy and inverse transform to get LOG RETURNS
        predictions_scaled = predictions_scaled.cpu().numpy()
        predicted_returns = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        
        # Convert log returns to actual prices
        if last_sequence_only:
            # predicted_return is a log return: log(price_tomorrow / price_today)
            # To get price_tomorrow: price_today * exp(log_return)
            predicted_price = last_close * np.exp(predicted_returns[0])
            
            return {
                'predicted_return': predicted_returns[0],
                'predicted_price': predicted_price,
                'last_close': last_close,
                'return_pct': (np.exp(predicted_returns[0]) - 1) * 100  # Convert to percentage
            }
        else:
            # Multiple predictions
            predicted_prices = []
            for i, log_return in enumerate(predicted_returns):
                if last_sequence_only:
                    base_price = last_close
                else:
                    base_price = close_prices[i]
                predicted_price = base_price * np.exp(log_return)
                predicted_prices.append(predicted_price)
            
            return {
                'predicted_returns': predicted_returns,
                'predicted_prices': np.array(predicted_prices),
                'base_close_prices': np.array(close_prices) if not last_sequence_only else last_close
            }
    
    def save_model(self, ticker: str, metadata: Optional[dict] = None):
        """Save model weights and configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"enhanced_lstm_{ticker}_{timestamp}"
        
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
            'bidirectional': self.bidirectional,
            'use_layer_norm': self.use_layer_norm,
            'use_residual': self.use_residual,
            'feature_columns': self.feature_columns,
            'ticker': ticker,
            'timestamp': timestamp,
            'device': str(self.device),
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm
        }
        
        if metadata:
            config.update(metadata)
        
        config_path = os.path.join(save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Save scalers
        scaler_data = {
            'feature_scaler_data_min_': self.feature_scaler.data_min_.tolist(),
            'feature_scaler_data_max_': self.feature_scaler.data_max_.tolist(),
            'feature_scaler_data_range_': self.feature_scaler.data_range_.tolist(),
            'target_scaler_data_min_': self.target_scaler.data_min_.tolist(),
            'target_scaler_data_max_': self.target_scaler.data_max_.tolist(),
            'target_scaler_data_range_': self.target_scaler.data_range_.tolist(),
        }
        
        scaler_path = os.path.join(save_path, "scalers.json")
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f, indent=4)
        
        print(f"Enhanced model saved to: {save_path}")
        return save_path
    
    def _filter_loss_config(self, config: dict) -> dict:
        """
        Filter loss configuration parameters based on loss type to avoid unexpected arguments
        """
        loss_type = config.get('loss_type', 'standard')
        
        # Base parameters that all losses accept
        base_params = {
            'loss_type': loss_type
        }
        
        if loss_type == 'standard':
            # DirectionalLoss parameters
            allowed_params = ['price_weight', 'direction_weight', 'price_loss_type', 'direction_threshold']
            filtered_config = base_params.copy()
            for param in allowed_params:
                if param in config:
                    filtered_config[param] = config[param]
            return filtered_config
        
        elif loss_type == 'focal':
            # FocalDirectionalLoss parameters
            allowed_params = ['price_weight', 'direction_weight', 'focal_alpha', 'focal_gamma']
            filtered_config = base_params.copy()
            for param in allowed_params:
                if param in config:
                    filtered_config[param] = config[param]
            return filtered_config
        
        elif loss_type == 'ranking':
            # RankingDirectionalLoss parameters
            allowed_params = ['price_weight', 'ranking_weight']
            filtered_config = base_params.copy()
            for param in allowed_params:
                if param in config:
                    filtered_config[param] = config[param]
            return filtered_config
        
        elif loss_type == 'adaptive':
            # AdaptiveDirectionalLoss parameters
            allowed_params = ['initial_price_weight', 'target_direction_accuracy', 'adaptation_rate']
            filtered_config = base_params.copy()
            for param in allowed_params:
                if param in config:
                    filtered_config[param] = config[param]
            return filtered_config
        
        else:
            # Fallback to base parameters
            print(f"Warning: Unknown loss type '{loss_type}', using base parameters only")
            return base_params
    
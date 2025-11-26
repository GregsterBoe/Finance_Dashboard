"""
Selective LSTM Stock Predictor with Abstention Capability

This model can learn WHEN to make predictions vs. when to abstain.
It uses a dual-head architecture:
- Price prediction head: Predicts the stock price return
- Confidence head: Learns when predictions are likely to be accurate

Key innovation: The model is trained to abstain on difficult-to-predict days,
improving overall accuracy on days where it does make predictions.

=======================================================================
IMPROVED VERSION HIGHLIGHTS:
- [MODEL]   Added torch.sigmoid to price_prediction output to match (0, 1) scaler
- [LOSS]    Replaced unstable normalized_error with direct F.l1_loss
- [LOSS]    Simplified loss logic to be much more readable
- [TRAIN]   Clarified metric logging (total vs. selective accuracy)
=======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Dict
import os
import json
from datetime import datetime

# Mock services for standalone execution
class MockFeatureService:
    def prepare_for_lstm(self, df, feature_set):
        # Create some dummy features for testing
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['RSI_14'] = df['Close'].pct_change().rolling(14).mean() # Simplified
        df['Momentum'] = df['Close'].pct_change(5)
        feature_cols = ['SMA_10', 'RSI_14', 'Momentum']
        return df, feature_cols

def get_feature_service():
    return MockFeatureService()

class FeatureSet:
    LSTM = "lstm"


class SelectiveLSTMModel(nn.Module):
    """
    Dual-head LSTM: One head for price prediction, one for confidence/abstention
    """
    def __init__(self, input_size: int,
                hidden_size: int = 64, 
                num_layers: int = 2,
                dropout: float = 0.2, 
                bidirectional: bool = True,
                use_layer_norm: bool = True,
                use_residual: bool = True):
        
        super(SelectiveLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        if use_residual and input_size != lstm_output_size:
            self.residual_proj = nn.Linear(input_size, lstm_output_size)
        else:
            self.residual_proj = None
        
        self.dropout = nn.Dropout(dropout)
        
        # ========================================================================
        # PRICE PREDICTION HEAD
        # ========================================================================
        self.price_fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.price_fc2 = nn.Linear(hidden_size, 1)
        
        if use_layer_norm:
            self.price_layer_norm = nn.LayerNorm(hidden_size)
        
        # ========================================================================
        # CONFIDENCE/ABSTENTION HEAD
        # ========================================================================
        self.confidence_fc1 = nn.Linear(lstm_output_size, hidden_size // 2)
        self.confidence_fc2 = nn.Linear(hidden_size // 2, 1)
        
        if use_layer_norm:
            self.confidence_layer_norm = nn.LayerNorm(hidden_size // 2)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass with dual outputs
        """
        batch_size = x.size(0)
        
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        if self.bidirectional:
            last_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            last_hidden = h_n[-1, :, :]
        
        if self.use_layer_norm:
            last_hidden = self.layer_norm(last_hidden)
        
        if self.use_residual:
            residual = x[:, -1, :]
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            last_hidden = last_hidden + residual
        
        # ========================================================================
        # PRICE PREDICTION HEAD
        # ========================================================================
        price_out = self.price_fc1(last_hidden)
        price_out = self.relu(price_out)
        
        if self.use_layer_norm:
            price_out = self.price_layer_norm(price_out)
        
        price_out = self.dropout(price_out)
        price_logit = self.price_fc2(price_out)
        
        # ---
        # **FIXED**: Apply sigmoid activation
        # This constrains the output to (0, 1), matching your target scaler.
        # ---
        price_prediction = torch.sigmoid(price_logit)
        
        # ========================================================================
        # CONFIDENCE HEAD
        # ========================================================================
        conf_out = self.confidence_fc1(last_hidden)
        conf_out = self.relu(conf_out)
        
        if self.use_layer_norm:
            conf_out = self.confidence_layer_norm(conf_out)
        
        conf_out = self.dropout(conf_out)
        confidence_logit = self.confidence_fc2(conf_out)
        confidence_score = torch.sigmoid(confidence_logit)
        
        return price_prediction, confidence_score


class SelectiveLoss(nn.Module):
    """
    **High-Precision Loss Function**

    This loss is designed to maximize precision (correctness) at the
    cost of coverage (recall). It has two independent parts:

    1.  **Price Loss (L1/MAE)**:
        This is a standard loss that trains the price prediction head
        to be as accurate as possible on *all* samples.

    2.  **Confidence Loss (MSE)**:
        This trains the confidence head to solve a binary classification
        problem: "Will the price head's direction be correct?"
        - Target = 1.0 if direction is correct
        - Target = 0.0 if direction is wrong
    
    This decouples the tasks: the price head learns to predict price,
    and the confidence head learns to predict *when the price head is right*.
    """
    def __init__(self, confidence_weight: float = 1.0):
        """
        Args:
            confidence_weight: How much to weigh the confidence loss
                               relative to the price prediction loss.
        """
        super(SelectiveLoss, self).__init__()
        self.confidence_weight = confidence_weight
        
    def forward(self, price_pred, confidence, actual_price, prev_price):
        """
        Calculate the decoupled loss.
        """
        
        # ========================================================================
        # 1. PRICE PREDICTION LOSS (Trains the price head)
        # ========================================================================
        # Standard L1 loss to train the price head to be accurate.
        # This is calculated on *all* samples.
        price_loss = F.l1_loss(price_pred, actual_price)
        
        # ========================================================================
        # 2. CONFIDENCE CALIBRATION LOSS (Trains the confidence head)
        # ========================================================================
        
        # Determine the "ground truth" for confidence: was the direction correct?
        # We must detach price_pred and prev_price here so that this
        # calculation does not send gradients back to the price head.
        pred_direction = (price_pred.detach() > prev_price.detach()).float()
        actual_direction = (actual_price.detach() > prev_price.detach()).float()
        
        # The target for confidence is 1.0 if correct, 0.0 if wrong.
        target_confidence = (pred_direction == actual_direction).float()
        
        # Train the confidence head to predict this target.
        # We use MSELoss (confidence vs. 0.0/1.0 target).
        confidence_loss = F.mse_loss(confidence, target_confidence)
        
        # ========================================================================
        # TOTAL LOSS
        # ========================================================================
        total_loss = price_loss + (self.confidence_weight * confidence_loss)
        
        # Return loss and metrics for monitoring
        metrics = {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'avg_target_confidence': target_confidence.mean().item() # This is the batch's total accuracy
        }
        
        return total_loss, metrics


class SelectiveLSTMPredictor:
    """
    Selective LSTM Stock Predictor
    Refactored for HIGH PRECISION, LOW COVERAGE.

    This model no longer tries to hit a 'target_coverage'.
    Instead, it learns to predict its own directional accuracy.
    You control coverage by setting a high 'confidence_threshold'
    at prediction time.
    """
    def __init__(self, 
                 sequence_length: int = 20, 
                 hidden_size: int = 64,
                 num_layers: int = 2, 
                 dropout: float = 0.2, 
                 learning_rate: float = 0.001, 
                 model_dir: str = 'selective_lstm_models',
                 bidirectional: bool = True, 
                 use_layer_norm: bool = True,
                 use_residual: bool = True, 
                 weight_decay: float = 1e-5,
                 gradient_clip_norm: float = 1.0,
                 # --- NEW/CHANGED PARAMETERS ---
                 confidence_weight: float = 1.0,
                 confidence_threshold: float = 0.8
                 # --- REMOVED PARAMETERS ---
                 # target_coverage (removed)
                 # coverage_weight (removed)
                 # direction_weight (removed)
                ):
        
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
        
        # Parameters for the new loss
        self.confidence_weight = confidence_weight
        
        # This is now *only* for prediction, not training
        self.confidence_threshold = confidence_threshold
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"High-Precision Selective LSTM Initialized.")
        print(f"Prediction Confidence Threshold: {self.confidence_threshold}")

    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for LSTM training, including previous prices for direction calculation"""
        X, y, y_prev = [], [], []
        
        # Start at sequence_length to have t-1 price for the first target y
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
            y_prev.append(targets[i-1])  # Previous day's price
        
        return np.array(X), np.array(y), np.array(y_prev)

    def train(self, df: pd.DataFrame, 
              epochs: int = 100, 
              batch_size: int = 32,
              validation_sequences: int = 50,
              early_stopping_patience: int = 20,
              use_validation: bool = True,
              lr_patience: int = 10,
              lr_factor: float = 0.5,
              min_lr: float = 1e-6) -> Dict:
        """
        Train the selective LSTM model
        """
        print(f"\n{'='*80}")
        print("TRAINING HIGH-PRECISION SELECTIVE LSTM")
        print(f"{'='*80}")
        
        # (This section is unchanged from your file)
        feature_service = get_feature_service()
        df_features, feature_cols = feature_service.prepare_for_lstm(
            df, 
            feature_set=FeatureSet.LSTM
        )
        self.feature_columns = feature_cols
        df_clean = df_features[self.feature_columns + ['Close']].dropna()
        if len(df_clean) < self.sequence_length + 2:
            raise ValueError("Not enough data to create even one sequence.")
        
        features = df_clean[self.feature_columns].values
        targets = df_clean['Close'].values.reshape(-1, 1)
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)
        features_scaled = self.feature_scaler.transform(features)
        targets_scaled = self.target_scaler.transform(targets)
        
        X, y, y_prev = self._create_sequences(features_scaled, targets_scaled)
        # (Data splitting logic is unchanged)
        if use_validation:
            val_size = min(validation_sequences, len(X) // 5)
            if val_size == 0 and len(X) > 1: val_size = 1
            if val_size == 0:
                print("Warning: Not enough data for validation split. Disabling validation.")
                use_validation = False
                X_train, y_train, y_prev_train = X, y, y_prev
                X_val = y_val = y_prev_val = None
            else:
                X_train, X_val = X[:-val_size], X[-val_size:]
                y_train, y_val = y[:-val_size], y[-val_size:]
                y_prev_train, y_prev_val = y_prev[:-val_size], y_prev[-val_size:]
        else:
            X_train, y_train, y_prev_train = X, y, y_prev
            X_val = y_val = y_prev_val = None
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_prev_train = torch.FloatTensor(y_prev_train).to(self.device)
        
        if use_validation:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            y_prev_val = torch.FloatTensor(y_prev_val).to(self.device)
        
        input_size = X_train.shape[2]
        self.model = SelectiveLSTMModel(
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
            optimizer, mode='min', factor=lr_factor, 
            patience=lr_patience, min_lr=min_lr
        )
        
        # --- (CRITERION IS UPDATED) ---
        criterion = SelectiveLoss(
            confidence_weight=self.confidence_weight
        )
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_price_loss': [], 'val_price_loss': [],
            'train_conf_loss': [], 'val_conf_loss': [],
            'train_total_dir_acc': [], 'val_selective_dir_acc': [],
            'val_selective_coverage': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            train_metrics_list = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                batch_y_prev = y_prev_train[i:i+batch_size]
                
                if len(batch_X) == 0: continue
                
                optimizer.zero_grad()
                price_pred, confidence = self.model(batch_X)
                
                # --- (Loss call is updated) ---
                loss, metrics = criterion(price_pred, confidence, batch_y, batch_y_prev)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                
                train_metrics_list.append(metrics)
            
            avg_train_metrics = {k: np.mean([m[k] for m in train_metrics_list]) for k in train_metrics_list[0].keys()}
            
            if use_validation:
                self.model.eval()
                val_selective_acc = 0.0
                val_selective_coverage = 0.0
                
                with torch.no_grad():
                    price_pred_val, confidence_val = self.model(X_val)
                    # --- (Loss call is updated) ---
                    val_loss, val_metrics = criterion(price_pred_val, confidence_val, y_val, y_prev_val)
                    
                    scheduler.step(val_loss)
                    
                    # --- (Validation logic is unchanged) ---
                    # We still use the *fixed* threshold to see how
                    # the model would perform in production.
                    high_conf_mask = (confidence_val > self.confidence_threshold).squeeze()
                    val_selective_coverage = high_conf_mask.float().mean().item()
                    
                    if high_conf_mask.sum() > 0:
                        price_pred_selective = price_pred_val[high_conf_mask]
                        y_val_selective = y_val[high_conf_mask]
                        y_prev_selective = y_prev_val[high_conf_mask]
                        
                        pred_dir = (price_pred_selective > y_prev_selective).float()
                        actual_dir = (y_val_selective > y_prev_selective).float()
                        val_selective_acc = (pred_dir == actual_dir).float().mean().item()
                    else:
                        val_selective_acc = 0.0 # No predictions made
                
                # --- (History logging is updated) ---
                history['train_loss'].append(avg_train_metrics['total_loss'])
                history['val_loss'].append(val_metrics['total_loss'])
                history['train_price_loss'].append(avg_train_metrics['price_loss'])
                history['val_price_loss'].append(val_metrics['price_loss'])
                history['train_conf_loss'].append(avg_train_metrics['confidence_loss'])
                history['val_conf_loss'].append(val_metrics['confidence_loss'])
                history['train_total_dir_acc'].append(avg_train_metrics['avg_target_confidence'])
                history['val_selective_dir_acc'].append(val_selective_acc)
                history['val_selective_coverage'].append(val_selective_coverage)
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                
                if (epoch + 1) % 10 == 0:
                    print(f"\nEpoch {epoch+1}/{epochs}")
                    print(f"  Train Loss: {avg_train_metrics['total_loss']:.6f} | Val Loss: {val_metrics['total_loss']:.6f}")
                    print(f"    - Price Loss (T/V): {avg_train_metrics['price_loss']:.4f} / {val_metrics['price_loss']:.4f}")
                    print(f"    - Conf. Loss (T/V): {avg_train_metrics['confidence_loss']:.4f} / {val_metrics['confidence_loss']:.4f}")
                    print(f"  Train Total Dir Acc: {avg_train_metrics['avg_target_confidence']*100:.1f}%")
                    print(f"  Val Selective Acc: {val_selective_acc*100:.1f}% (Coverage: {val_selective_coverage*100:.1f}%)")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                # (Early stopping logic is unchanged)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'best_model.pth'))
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'best_model.pth')))
                    break
            else:
                # No validation
                history['train_loss'].append(avg_train_metrics['total_loss'])
                history['learning_rate'].append(optimizer.param_groups[0]['lr'])
                
                if (epoch + 1) % 10 == 0:
                    print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train_metrics['total_loss']:.6f}")
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")
        
        return history

    # ---
    # The predict_with_confidence and predict methods are
    # unchanged from your original 'selective_lstm_model.py' file.
    # They already use self.confidence_threshold, which is now
    # the correct behavior.
    # ---
    def predict_with_confidence(self, df: pd.DataFrame, last_sequence_only: bool = True):
        """
        Make predictions with confidence scores
        (This function is unchanged)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        feature_service = get_feature_service()
        df_features, _ = feature_service.prepare_for_lstm(
            df,
            feature_set=FeatureSet.LSTM
        )
        original_indices = df_features.index
        df_clean = df_features[self.feature_columns + ['Close']].dropna()
        
        if len(df_clean) < self.sequence_length:
            print(f"Warning: Insufficient data for prediction. Need {self.sequence_length} rows, found {len(df_clean)}")
            return {
                'predicted_prices': np.array([]),
                'confidence_scores': np.array([]),
                'should_predict': np.array([]),
                'predicted_prices_selective': np.array([]),
                'coverage': 0.0
            }

        last_close = df_clean['Close'].iloc[-1]
        features_scaled = self.feature_scaler.transform(df_clean[self.feature_columns].values)
        
        if last_sequence_only:
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            prediction_indices = [df_clean.index[-1]] 
        else:
            num_sequences = len(features_scaled) - self.sequence_length + 1
            X = np.zeros((num_sequences, self.sequence_length, features_scaled.shape[1]))
            for i in range(num_sequences):
                X[i] = features_scaled[i:i+self.sequence_length]
            prediction_indices = df_clean.index[self.sequence_length-1:]
        
        X = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            price_pred_scaled, confidence = self.model(X)
        
        price_pred_scaled = price_pred_scaled.cpu().numpy()
        confidence = confidence.cpu().numpy()
        
        predicted_prices = self.target_scaler.inverse_transform(price_pred_scaled)
        
        # This is the key: it uses the fixed threshold
        should_predict = confidence > self.confidence_threshold
        
        predicted_prices_selective = predicted_prices.copy()
        predicted_prices_selective[~should_predict] = np.nan
        
        results_df = pd.DataFrame({
            'predicted_prices': predicted_prices.flatten(),
            'confidence_scores': confidence.flatten(),
            'should_predict': should_predict.flatten(),
            'predicted_prices_selective': predicted_prices_selective.flatten()
        }, index=prediction_indices)
        
        results_df = results_df.reindex(original_indices, fill_value=np.nan)
        
        if last_sequence_only:
            return {
                'predicted_prices': results_df['predicted_prices'].values,
                'confidence_scores': results_df['confidence_scores'].values,
                'should_predict': results_df['should_predict'].values,
                'predicted_prices_selective': results_df['predicted_prices_selective'].values,
                'last_close': last_close,
                'coverage': should_predict.mean()
            }
        else:
            return results_df

    def predict(self, df: pd.DataFrame, last_sequence_only: bool = True):
        """
        Standard predict method
        (This function is unchanged)
        """
        result = self.predict_with_confidence(df, last_sequence_only)
        
        if last_sequence_only:
            selective_price = result['predicted_prices_selective'][-1]
            return np.array([[selective_price]]) # Return NaN or value
        else:
            return result['predicted_prices_selective'].values.reshape(-1, 1)
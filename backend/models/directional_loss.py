"""
Custom loss functions for stock price prediction with directional accuracy optimization.

This module provides various loss functions that can improve directional accuracy
while maintaining price prediction capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DirectionalLoss(nn.Module):
    """
    Combined loss that optimizes for both price accuracy and directional accuracy.
    
    This loss function helps the model learn not just accurate prices, but also
    correct directional movements (up/down) which is crucial for trading decisions.
    """
    
    def __init__(self, 
                 price_weight: float = 0.7, 
                 direction_weight: float = 0.3,
                 price_loss_type: str = 'mse',
                 direction_threshold: float = 0.0):
        """
        Args:
            price_weight: Weight for price prediction loss (0-1)
            direction_weight: Weight for directional accuracy loss (0-1)
            price_loss_type: Type of price loss ('mse', 'mae', 'huber')
            direction_threshold: Threshold for direction calculation (default 0.0)
        """
        super(DirectionalLoss, self).__init__()
        
        assert abs(price_weight + direction_weight - 1.0) < 1e-6, \
            "price_weight + direction_weight must equal 1.0"
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.direction_threshold = direction_threshold
        
        # Price loss function
        if price_loss_type == 'mse':
            self.price_loss_fn = nn.MSELoss()
        elif price_loss_type == 'mae':
            self.price_loss_fn = nn.L1Loss()
        elif price_loss_type == 'huber':
            self.price_loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported price_loss_type: {price_loss_type}")
        
        # Direction loss function
        self.direction_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, 
                predicted_prices: torch.Tensor, 
                actual_prices: torch.Tensor, 
                previous_prices: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Calculate combined price and directional loss.
        
        Args:
            predicted_prices: Predicted close prices [batch_size, 1]
            actual_prices: Actual close prices [batch_size, 1]
            previous_prices: Previous day's close prices [batch_size, 1]
            
        Returns:
            total_loss: Combined weighted loss
            loss_components: Dict with breakdown of loss components
        """
        # Price loss (standard MSE/MAE/Huber)
        price_loss = self.price_loss_fn(predicted_prices, actual_prices)
        
        # Directional accuracy loss
        predicted_direction = (predicted_prices > previous_prices + self.direction_threshold).float()
        actual_direction = (actual_prices > previous_prices + self.direction_threshold).float()
        
        # Use BCEWithLogitsLoss for numerical stability
        # Convert boolean directions to logits space
        predicted_logits = torch.log(predicted_direction + 1e-8) - torch.log(1 - predicted_direction + 1e-8)
        direction_loss = self.direction_loss_fn(predicted_logits, actual_direction)
        
        # Combined loss
        total_loss = (self.price_weight * price_loss + 
                     self.direction_weight * direction_loss)
        
        # Calculate directional accuracy for monitoring
        with torch.no_grad():
            direction_accuracy = (predicted_direction == actual_direction).float().mean().item()
        
        loss_components = {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'direction_loss': direction_loss.item(),
            'direction_accuracy': direction_accuracy * 100  # As percentage
        }
        
        return total_loss, loss_components


class FocalDirectionalLoss(nn.Module):
    """
    Focal loss variant that focuses more on hard-to-predict directional changes.
    
    This loss puts more emphasis on examples where the model struggles with
    directional prediction, potentially improving overall directional accuracy.
    """
    
    def __init__(self, 
                 price_weight: float = 0.6,
                 direction_weight: float = 0.4,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            price_weight: Weight for price prediction loss
            direction_weight: Weight for directional loss
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter (focus on hard examples)
        """
        super(FocalDirectionalLoss, self).__init__()
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.price_loss_fn = nn.MSELoss()
    
    def forward(self, predicted_prices, actual_prices, previous_prices):
        """Calculate focal directional loss"""
        
        # Standard price loss
        price_loss = self.price_loss_fn(predicted_prices, actual_prices)
        
        # Directional predictions
        predicted_direction = (predicted_prices > previous_prices).float()
        actual_direction = (actual_prices > previous_prices).float()
        
        # Focal loss for direction
        bce_loss = F.binary_cross_entropy(predicted_direction, actual_direction, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of correct classification
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        direction_loss = focal_loss.mean()
        
        total_loss = self.price_weight * price_loss + self.direction_weight * direction_loss
        
        # Monitoring metrics
        with torch.no_grad():
            direction_accuracy = (predicted_direction == actual_direction).float().mean().item()
        
        loss_components = {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'direction_loss': direction_loss.item(),
            'direction_accuracy': direction_accuracy * 100
        }
        
        return total_loss, loss_components


class RankingDirectionalLoss(nn.Module):
    """
    Loss that optimizes for relative ranking of predictions within a batch.
    
    This encourages the model to correctly rank which stocks will perform
    better relative to others, which can improve directional accuracy.
    """
    
    def __init__(self, price_weight: float = 0.5, ranking_weight: float = 0.5):
        super(RankingDirectionalLoss, self).__init__()
        self.price_weight = price_weight
        self.ranking_weight = ranking_weight
        self.price_loss_fn = nn.MSELoss()
    
    def forward(self, predicted_prices, actual_prices, previous_prices):
        """Calculate ranking-based directional loss"""
        
        # Standard price loss
        price_loss = self.price_loss_fn(predicted_prices, actual_prices)
        
        # Calculate returns
        predicted_returns = (predicted_prices - previous_prices) / previous_prices
        actual_returns = (actual_prices - previous_prices) / previous_prices
        
        # Ranking loss (margin ranking loss)
        batch_size = predicted_returns.size(0)
        ranking_loss = 0
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # If actual_returns[i] > actual_returns[j], then predicted_returns[i] should be > predicted_returns[j]
                if actual_returns[i] != actual_returns[j]:  # Avoid ties
                    target = torch.sign(actual_returns[i] - actual_returns[j])
                    predicted_diff = predicted_returns[i] - predicted_returns[j]
                    ranking_loss += F.margin_ranking_loss(
                        predicted_diff.unsqueeze(0), 
                        torch.zeros_like(predicted_diff.unsqueeze(0)), 
                        target.unsqueeze(0),
                        margin=0.01
                    )
                    count += 1
        
        if count > 0:
            ranking_loss = ranking_loss / count
        else:
            ranking_loss = torch.tensor(0.0, device=predicted_prices.device, requires_grad=True)
        
        total_loss = self.price_weight * price_loss + self.ranking_weight * ranking_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'ranking_loss': ranking_loss.item()
        }
        
        return total_loss, loss_components


class AdaptiveDirectionalLoss(nn.Module):
    """
    Adaptive loss that adjusts the weight between price and direction loss
    based on the current directional accuracy performance.
    """
    
    def __init__(self, 
                 initial_price_weight: float = 0.7,
                 target_direction_accuracy: float = 60.0,
                 adaptation_rate: float = 0.01):
        """
        Args:
            initial_price_weight: Starting weight for price loss
            target_direction_accuracy: Target directional accuracy (%)
            adaptation_rate: How fast to adapt weights
        """
        super(AdaptiveDirectionalLoss, self).__init__()
        
        self.price_weight = initial_price_weight
        self.direction_weight = 1.0 - initial_price_weight
        self.target_direction_accuracy = target_direction_accuracy
        self.adaptation_rate = adaptation_rate
        
        self.price_loss_fn = nn.MSELoss()
        self.direction_loss_fn = nn.BCEWithLogitsLoss()
        
        # Track recent directional accuracy
        self.recent_accuracy = []
        self.window_size = 100
    
    def forward(self, predicted_prices, actual_prices, previous_prices):
        """Calculate adaptive directional loss"""
        
        # Standard losses
        price_loss = self.price_loss_fn(predicted_prices, actual_prices)
        
        predicted_direction = (predicted_prices > previous_prices).float()
        actual_direction = (actual_prices > previous_prices).float()
        
        predicted_logits = torch.log(predicted_direction + 1e-8) - torch.log(1 - predicted_direction + 1e-8)
        direction_loss = self.direction_loss_fn(predicted_logits, actual_direction)
        
        # Calculate current directional accuracy
        with torch.no_grad():
            current_accuracy = (predicted_direction == actual_direction).float().mean().item() * 100
            self.recent_accuracy.append(current_accuracy)
            
            # Keep only recent history
            if len(self.recent_accuracy) > self.window_size:
                self.recent_accuracy.pop(0)
            
            # Adapt weights based on performance
            if len(self.recent_accuracy) >= 10:  # Need some history
                avg_accuracy = np.mean(self.recent_accuracy[-10:])  # Last 10 batches
                
                if avg_accuracy < self.target_direction_accuracy:
                    # Increase focus on directional loss
                    self.direction_weight = min(0.8, self.direction_weight + self.adaptation_rate)
                else:
                    # Can focus more on price accuracy
                    self.direction_weight = max(0.2, self.direction_weight - self.adaptation_rate)
                
                self.price_weight = 1.0 - self.direction_weight
        
        total_loss = self.price_weight * price_loss + self.direction_weight * direction_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'price_loss': price_loss.item(),
            'direction_loss': direction_loss.item(),
            'direction_accuracy': current_accuracy,
            'price_weight': self.price_weight,
            'direction_weight': self.direction_weight
        }
        
        return total_loss, loss_components


# Utility function to create loss functions
def create_directional_loss(loss_type: str = 'standard', **kwargs):
    """
    Factory function to create directional loss functions.
    
    Args:
        loss_type: Type of loss ('standard', 'focal', 'ranking', 'adaptive')
        **kwargs: Additional arguments for the specific loss function
    
    Returns:
        Instantiated loss function
    """
    if loss_type == 'standard':
        return DirectionalLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalDirectionalLoss(**kwargs)
    elif loss_type == 'ranking':
        return RankingDirectionalLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveDirectionalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
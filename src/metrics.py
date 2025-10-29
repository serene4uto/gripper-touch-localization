"""
Metrics module for gripper touch localization model evaluation.
Centralizes all metric calculations and evaluation functions.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        predictions: Model predictions (tensor)
        targets: Ground truth targets (tensor)
        
    Returns:
        MAE value (float)
    """
    return torch.mean(torch.abs(predictions - targets)).item()


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        predictions: Model predictions (tensor)
        targets: Ground truth targets (tensor)
        
    Returns:
        MSE value (float)
    """
    return torch.mean((predictions - targets) ** 2).item()


def calculate_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        predictions: Model predictions (tensor)
        targets: Ground truth targets (tensor)
        
    Returns:
        RMSE value (float)
    """
    mse = calculate_mse(predictions, targets)
    return np.sqrt(mse)


def calculate_r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        predictions: Model predictions (tensor)
        targets: Ground truth targets (tensor)
        
    Returns:
        R² score (float)
    """
    # Convert to numpy for easier calculation
    y_pred = predictions.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    
    # Calculate R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def calculate_mape(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        predictions: Model predictions (tensor)
        targets: Ground truth targets (tensor)
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value (float)
    """
    # Convert to numpy for easier calculation
    y_pred = predictions.detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    
    # Avoid division by zero
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return float(mape)


def evaluate_model_metrics(model: torch.nn.Module, 
                          dataloader: torch.utils.data.DataLoader, 
                          device: torch.device,
                          criterion: torch.nn.Module = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)
        
    Returns:
        Dictionary containing all calculated metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Store predictions and targets
            all_predictions.append(outputs)
            all_targets.append(batch_y)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = {
        'mae': calculate_mae(all_predictions, all_targets),
        'mse': calculate_mse(all_predictions, all_targets),
        'rmse': calculate_rmse(all_predictions, all_targets),
        'r2_score': calculate_r2_score(all_predictions, all_targets),
        'mape': calculate_mape(all_predictions, all_targets)
    }
    
    # Add loss if criterion was provided
    if criterion is not None:
        metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        if metric_name == 'loss':
            print(f"{metric_name.upper():<12}: {value:.6f}")
        elif metric_name == 'r2_score':
            print(f"{metric_name.upper():<12}: {value:.4f}")
        elif metric_name == 'mape':
            print(f"{metric_name.upper():<12}: {value:.2f}%")
        else:
            print(f"{metric_name.upper():<12}: {value:.4f}")
    
    print(f"{'='*50}")


def plot_predictions_vs_targets(predictions: torch.Tensor, 
                               targets: torch.Tensor, 
                               title: str = "Predictions vs Targets",
                               save_path: str = None):
    """
    Plot predictions against targets for visual evaluation.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Convert to numpy
    y_pred = predictions.detach().cpu().numpy().flatten()
    y_true = targets.detach().cpu().numpy().flatten()
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display R²
    r2 = calculate_r2_score(predictions, targets)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_residuals(predictions: torch.Tensor, 
                  targets: torch.Tensor, 
                  title: str = "Residuals Plot",
                  save_path: str = None):
    """
    Plot residuals (predictions - targets) for error analysis.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Convert to numpy
    y_pred = predictions.detach().cpu().numpy().flatten()
    y_true = targets.detach().cpu().numpy().flatten()
    
    # Calculate residuals
    residuals = y_pred - y_true
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs predictions
    ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predictions')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def compare_models_metrics(metrics_dict: Dict[str, Dict[str, float]], 
                          title: str = "Model Comparison"):
    """
    Compare metrics across multiple models.
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        title: Title for the comparison
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Get all metric names
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    
    # Print header
    print(f"{'Model':<20}", end="")
    for metric in sorted(all_metrics):
        print(f"{metric.upper():<12}", end="")
    print()
    
    print("-" * (20 + 12 * len(all_metrics)))
    
    # Print metrics for each model
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<20}", end="")
        for metric in sorted(all_metrics):
            value = metrics.get(metric, 0.0)
            if metric == 'r2_score':
                print(f"{value:<12.4f}", end="")
            elif metric == 'mape':
                print(f"{value:<12.2f}", end="")
            else:
                print(f"{value:<12.4f}", end="")
        print()
    
    print(f"{'='*60}")

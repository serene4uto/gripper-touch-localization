#!/usr/bin/env python3
"""
Gripper Touch Localization Model Evaluation Script
Evaluates a trained ConvLSTM model on test data and provides comprehensive metrics
"""

import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data import load_data, get_dataloaders
from src.model import ConvLSTMRegressor
from src.trainer import evaluate_model
from src.metrics import (
    calculate_mae, calculate_mse, calculate_rmse, 
    calculate_r2_score, calculate_mape, print_metrics_summary
)
from src.config import load_hyperparams

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained ConvLSTM model')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--config-path', type=str, 
                        help='Path to model config file (if different from training)')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='dataset/SizeTrain.csv',
                        help='Path to evaluation data CSV file')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to CSV file')
    parser.add_argument('--plot-results', action='store_true',
                        help='Generate prediction plots')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto)')
    
    return parser.parse_args()

def load_model(model_path, config_path=None, device='cpu'):
    """Load trained model from checkpoint"""
    print(f"📂 Loading model from: {model_path}")
    
    # Load model state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with metadata
        state_dict = checkpoint['model_state_dict']
        print(f"   Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # Just state dict
        state_dict = checkpoint
        print("   Loaded state dict directly")
    
    # Load config if provided
    if config_path and os.path.exists(config_path):
        hyperparams = load_hyperparams(config_path)
    else:
        # Try to infer from model path
        config_candidates = [
            os.path.join(os.path.dirname(model_path), 'config.yaml'),
            'configs/train/default_cfg.yaml'
        ]
        for candidate in config_candidates:
            if os.path.exists(candidate):
                hyperparams = load_hyperparams(candidate)
                print(f"   Using config: {candidate}")
                break
        else:
            raise FileNotFoundError("Could not find config file. Please specify --config-path")
    
    # Create model
    model = ConvLSTMRegressor(
        input_channels=2,
        hidden_channels=hyperparams['hidden_channels'],
        kernel_size=tuple(hyperparams['kernel_size']),
        dropout=hyperparams['dropout']
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"   Model loaded successfully on {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, hyperparams

def evaluate_comprehensive(model, test_loader, device='cpu'):
    """Comprehensive model evaluation with all metrics"""
    print("🔍 Running comprehensive evaluation...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            
            # Calculate loss
            criterion = torch.nn.MSELoss()
            loss = criterion(outputs.squeeze(), batch_labels)
            total_loss += loss.item()
            
            all_predictions.append(outputs.squeeze().cpu())
            all_targets.append(batch_labels.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate comprehensive metrics
    metrics = {
        'mse': calculate_mse(all_predictions, all_targets),
        'mae': calculate_mae(all_predictions, all_targets),
        'rmse': calculate_rmse(all_predictions, all_targets),
        'r2_score': calculate_r2_score(all_predictions, all_targets),
        'mape': calculate_mape(all_predictions, all_targets),
        'loss': total_loss / len(test_loader)
    }
    
    return metrics, all_predictions, all_targets

def plot_predictions(predictions, targets, save_path=None):
    """Plot predictions vs targets and residuals"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Predictions vs Targets scatter plot
    axes[0].scatter(targets.numpy(), predictions.numpy(), alpha=0.6, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title('Predictions vs True Values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = (targets - predictions).numpy()
    axes[1].scatter(predictions.numpy(), residuals, alpha=0.6, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predictions')
    axes[1].set_ylabel('Residuals (True - Predicted)')
    axes[1].set_title('Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {save_path}")
    
    plt.show()

def save_predictions(predictions, targets, save_path):
    """Save predictions to CSV file"""
    import pandas as pd
    
    results_df = pd.DataFrame({
        'true_values': targets.numpy(),
        'predictions': predictions.numpy(),
        'residuals': (targets - predictions).numpy(),
        'absolute_error': torch.abs(targets - predictions).numpy()
    })
    
    results_df.to_csv(save_path, index=False)
    print(f"💾 Predictions saved to: {save_path}")

def save_evaluation_report(metrics, model_info, output_dir):
    """Save comprehensive evaluation report"""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_info['model_path']}\n")
        f.write(f"Config Path: {model_info.get('config_path', 'N/A')}\n")
        f.write(f"Device: {model_info['device']}\n")
        f.write(f"Evaluation Samples: {model_info['eval_samples']}\n")
        f.write("\n")
        
        f.write("EVALUATION METRICS:\n")
        f.write("-" * 20 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.6f}\n")
        
        f.write("\n")
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 18 + "\n")
        for key, value in model_info['hyperparams'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n")
        f.write("=" * 60 + "\n")
    
    print(f"📋 Evaluation report saved to: {report_path}")

def main():
    args = parse_args()
    
    print("🔍 Starting Model Evaluation...")
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, hyperparams = load_model(args.model_path, args.config_path, device)
    
    # Load evaluation data
    print("📊 Loading evaluation data...")
    X_train, X_test, y_train, y_test, scaler = load_data(
        csv_path=args.data_path, 
        test_size=0.0  # Use all data for evaluation
    )
    print(f"Evaluation samples: {len(X_test)}")
    
    # Create evaluation dataloader
    _, eval_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=hyperparams['batch_size']
    )
    
    # Run comprehensive evaluation
    metrics, predictions, targets = evaluate_comprehensive(model, eval_loader, device)
    
    # Print results
    print("\n📈 Evaluation Results:")
    print("=" * 40)
    print_metrics_summary(metrics)
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        save_predictions(predictions, targets, predictions_path)
    
    # Generate plots if requested
    if args.plot_results:
        plots_path = os.path.join(args.output_dir, 'evaluation_plots.png')
        plot_predictions(predictions, targets, plots_path)
    
    # Save evaluation report
    model_info = {
        'model_path': args.model_path,
        'config_path': args.config_path,
        'device': str(device),
        'eval_samples': len(X_test),
        'hyperparams': hyperparams
    }
    save_evaluation_report(metrics, model_info, args.output_dir)
    
    print(f"\n✅ Evaluation Complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

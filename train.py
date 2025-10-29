#!/usr/bin/env python3
"""
Gripper Touch Localization Training Script
PyTorch implementation of ConvLSTM for contact area prediction
"""

import torch
import time
import os
import argparse
import shutil
from datetime import datetime
from src.data import load_data, get_dataloaders
from src.model import ConvLSTMRegressor
from src.trainer import train_model, plot_losses, evaluate_model, resume_training
from src.config import load_hyperparams, update_hyperparams_from_args, print_hyperparams

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ConvLSTM for gripper touch localization')
    
    # Config file argument
    parser.add_argument('--train-config', type=str, default='configs/train/default_cfg.yaml',
                        help='Path to YAML training configuration file (default: /workspaces/gripper-touch-localization/config/train/default_cfg.yaml)')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='dataset/SizeTrain.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set size (default: 0.3)')
    
    # Experiment arguments
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: exp_YYYYMMDD_HHMMSS)')
    
    # Note: Model and training hyperparameters are now managed via config.yaml
    # Use --config to specify a different config file
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto) (default: auto)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable loss plotting')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='Directory for TensorBoard logs (default: runs)')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    
    # Checkpoint arguments
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints, overridden by log-dir if TensorBoard enabled)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 Starting Gripper Touch Localization Training...")
    
    # Load hyperparameters from config file
    hyperparams = load_hyperparams(args.train_config)
    
    # Override with command line arguments if provided
    hyperparams = update_hyperparams_from_args(hyperparams, args)
    
    # Print hyperparameters
    print_hyperparams(hyperparams)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print("📊 Loading data...")
    X_train, X_test, y_train, y_test, scaler = load_data(
        csv_path=args.data_path, 
        test_size=hyperparams['test_size']
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create dataloaders
    train_loader, test_loader = get_dataloaders(
        X_train, X_test, y_train, y_test, 
        batch_size=hyperparams['batch_size']
    )
    
    # Create model
    print("🧠 Creating model...")
    model = ConvLSTMRegressor(
        input_channels=2,
        hidden_channels=hyperparams['hidden_channels'],
        kernel_size=tuple(hyperparams['kernel_size']),
        dropout=hyperparams['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up experiment name
    if args.exp_name is None:
        exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = args.exp_name
    
    # Train model
    print(f"🏋️ Training model... (Experiment: {exp_name})")
    start_time = time.time()
    
    # Set up TensorBoard logging
    if args.no_tensorboard:
        log_dir = None
        logs_dir = None
        checkpoint_dir = args.save_dir
        # Copy config to save directory when TensorBoard is disabled
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_dest = os.path.join(checkpoint_dir, 'config.yaml')
        shutil.copy2(args.train_config, config_dest)
        print(f"📄 Config saved to: {config_dest}")
    else:
        log_dir = os.path.join(args.log_dir, exp_name)
        checkpoint_dir = os.path.join(log_dir, 'checkpoint')
        # Create logs subdirectory for TensorBoard events
        logs_dir = os.path.join(log_dir, 'logs')
        
        # Create experiment directory and copy config file
        os.makedirs(log_dir, exist_ok=True)
        config_dest = os.path.join(log_dir, 'config.yaml')
        shutil.copy2(args.train_config, config_dest)
        print(f"📄 Config saved to: {config_dest}")
    
    # Check if resuming from checkpoint
    if args.resume:
        train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s, train_mapes, val_mapes, best_metrics = resume_training(
            model, train_loader, test_loader, args.resume,
            epochs=hyperparams['epochs'], lr=hyperparams['learning_rate'], device=device,
            log_dir=logs_dir, weight_decay=hyperparams['weight_decay'],
            save_dir=checkpoint_dir
        )
    else:
        train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s, train_mapes, val_mapes, best_metrics = train_model(
            model, train_loader, test_loader, 
            epochs=hyperparams['epochs'], lr=hyperparams['learning_rate'], device=device,
            log_dir=logs_dir, weight_decay=hyperparams['weight_decay'],
            save_dir=checkpoint_dir
        )
    
    training_time = time.time() - start_time
    print(f"⏱️ Training time: {training_time:.2f} seconds")
    
    # Evaluate
    print("📈 Evaluating model...")
    test_loss, test_mae = evaluate_model(model, test_loader, device)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Print best results summary
    print("\n🏆 Best Results Summary:")
    print(f"   Best Validation Loss: {best_metrics['best_val_loss']:.4f} (Epoch {best_metrics['best_epoch_loss']})")
    print(f"   Best Validation MAE:  {best_metrics['best_val_mae']:.4f} (Epoch {best_metrics['best_epoch_mae']})")
    print(f"   Best Validation RMSE: {best_metrics['best_val_rmse']:.4f} (Epoch {best_metrics['best_epoch_rmse']})")
    print(f"   Best Validation R²:   {best_metrics['best_val_r2']:.4f} (Epoch {best_metrics['best_epoch_r2']})")
    print(f"   Best Validation MAPE: {best_metrics['best_val_mape']:.4f} (Epoch {best_metrics['best_epoch_mape']})")
    
    # Plot results
    if not args.no_plot:
        plot_losses(train_losses, val_losses, train_maes, val_maes, 
                   train_rmses, val_rmses, train_r2s, val_r2s, train_mapes, val_mapes)
    
    # TensorBoard info
    if log_dir:
        print(f"📊 TensorBoard logs saved to: {logs_dir}")
        print(f"   View with: tensorboard --logdir {logs_dir}")
    
    # Save model to experiment directory
    if log_dir:
        # Save to experiment directory when TensorBoard is enabled
        model_path = os.path.join(log_dir, 'model.pth')
    else:
        # Save to checkpoint directory when TensorBoard is disabled
        model_path = os.path.join(checkpoint_dir, 'model.pth')
    
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved as '{model_path}'")
    
    # Save best results to text file
    results_dir = log_dir if log_dir else checkpoint_dir
    best_results_path = os.path.join(results_dir, 'best_results.txt')
    
    with open(best_results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Total Epochs: {hyperparams['epochs']}\n")
        f.write(f"Device: {device}\n")
        f.write("\n")
        f.write("BEST VALIDATION METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Validation Loss: {best_metrics['best_val_loss']:.6f} (Epoch {best_metrics['best_epoch_loss']})\n")
        f.write(f"Best Validation MAE:  {best_metrics['best_val_mae']:.6f} (Epoch {best_metrics['best_epoch_mae']})\n")
        f.write(f"Best Validation RMSE: {best_metrics['best_val_rmse']:.6f} (Epoch {best_metrics['best_epoch_rmse']})\n")
        f.write(f"Best Validation R²:   {best_metrics['best_val_r2']:.6f} (Epoch {best_metrics['best_epoch_r2']})\n")
        f.write(f"Best Validation MAPE: {best_metrics['best_val_mape']:.6f} (Epoch {best_metrics['best_epoch_mape']})\n")
        f.write("\n")
        f.write("FINAL TEST METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Test Loss (MSE): {test_loss:.6f}\n")
        f.write(f"Test MAE:        {test_mae:.6f}\n")
        f.write("\n")
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 15 + "\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write(f"Results saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
    
    print(f"📊 Best results saved to: {best_results_path}")
    
    print(f"✅ Training Complete! (Experiment: {exp_name})")

if __name__ == "__main__":
    main()

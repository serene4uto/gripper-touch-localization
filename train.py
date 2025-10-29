#!/usr/bin/env python3
"""
Gripper Touch Localization Training Script
PyTorch implementation of ConvLSTM for contact area prediction
"""

import torch
import time
import os
import argparse
from datetime import datetime
from src.data import load_data, get_dataloaders
from src.model import ConvLSTMRegressor
from src.trainer import train_model, plot_losses, evaluate_model
from src.config import load_hyperparams, update_hyperparams_from_args, print_hyperparams

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ConvLSTM for gripper touch localization')
    
    # Config file argument
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to YAML configuration file (default: config/config.yaml)')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='dataset/SizeTrain.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set size (default: 0.3)')
    
    # Experiment arguments
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: exp_YYYYMMDD_HHMMSS)')
    
    # Model arguments
    parser.add_argument('--hidden-channels', type=int, default=64,
                        help='Number of hidden channels in ConvLSTM (default: 64)')
    parser.add_argument('--kernel-size', type=int, nargs=2, default=[1, 3],
                        help='Kernel size for ConvLSTM (default: 1 3)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs (default: 250)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay for L2 regularization (default: 0.001)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cpu/cuda/auto) (default: auto)')
    parser.add_argument('--save-path', type=str, default='gripper_model.pth',
                        help='Path to save trained model (default: gripper_model.pth)')
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
    hyperparams = load_hyperparams(args.config)
    
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
        checkpoint_dir = args.save_dir
    else:
        log_dir = os.path.join(args.log_dir, exp_name)
        checkpoint_dir = os.path.join(log_dir, 'checkpoint')
        # Create logs subdirectory for TensorBoard events
        logs_dir = os.path.join(log_dir, 'logs')
    
    # Check if resuming from checkpoint
    if args.resume:
        from src.trainer import resume_training
        train_losses, val_losses, train_maes, val_maes = resume_training(
            model, train_loader, test_loader, args.resume,
            epochs=hyperparams['epochs'], lr=hyperparams['learning_rate'], device=device,
            log_dir=logs_dir, weight_decay=hyperparams['weight_decay'],
            save_dir=checkpoint_dir
        )
    else:
        train_losses, val_losses, train_maes, val_maes = train_model(
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
    
    # Plot results
    if not args.no_plot:
        plot_losses(train_losses, val_losses, train_maes, val_maes)
    
    # TensorBoard info
    if log_dir:
        print(f"📊 TensorBoard logs saved to: {logs_dir}")
        print(f"   View with: tensorboard --logdir {logs_dir}")
    
    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"💾 Model saved as '{args.save_path}'")
    
    print(f"✅ Training Complete! (Experiment: {exp_name})")

if __name__ == "__main__":
    main()

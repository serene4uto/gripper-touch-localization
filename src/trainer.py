import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .data import load_data, get_dataloaders
from .model import ConvLSTMRegressor
from .metrics import calculate_mae, calculate_mse, calculate_rmse, calculate_r2_score, calculate_mape

def train_model(model, train_loader, val_loader, epochs=250, lr=0.0001, device='cpu', 
                log_dir=None, weight_decay=0.001, save_dir=None, start_epoch=0):
    """Training loop with TensorBoard support and checkpointing (best and last only)"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # TensorBoard writer
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir)
    
    # Checkpoint tracking - track best metrics for all evaluation criteria
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    best_val_r2 = -float('inf')  # Maximize R²
    best_val_mape = float('inf')
    best_epoch_loss = 0
    best_epoch_mae = 0
    best_epoch_rmse = 0
    best_epoch_r2 = 0
    best_epoch_mape = 0
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_rmses = []
    val_rmses = []
    train_r2s = []
    val_r2s = []
    train_mapes = []
    val_mapes = []
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Training", leave=True)
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_rmse = 0.0
        train_r2 = 0.0
        train_mape = 0.0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                         leave=False, disable=len(train_loader) < 10)
        
        for batch_data, batch_labels in train_pbar:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            loss.backward()
            optimizer.step()
            
            # Calculate metrics for this batch
            batch_loss = loss.item()
            batch_mae = calculate_mae(outputs.squeeze(), batch_labels)
            batch_rmse = calculate_rmse(outputs.squeeze(), batch_labels)
            batch_r2 = calculate_r2_score(outputs.squeeze(), batch_labels)
            batch_mape = calculate_mape(outputs.squeeze(), batch_labels)
            
            train_loss += batch_loss
            train_mae += batch_mae
            train_rmse += batch_rmse
            train_r2 += batch_r2
            train_mape += batch_mape
            
            # Update training progress bar
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'MAE': f'{batch_mae:.4f}',
                'R²': f'{batch_r2:.4f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0
        val_r2 = 0.0
        val_mape = 0.0
        
        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                       leave=False, disable=len(val_loader) < 10)
        
        with torch.no_grad():
            for batch_data, batch_labels in val_pbar:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                
                # Calculate metrics for this batch
                batch_loss = loss.item()
                batch_mae = calculate_mae(outputs.squeeze(), batch_labels)
                batch_rmse = calculate_rmse(outputs.squeeze(), batch_labels)
                batch_r2 = calculate_r2_score(outputs.squeeze(), batch_labels)
                batch_mape = calculate_mape(outputs.squeeze(), batch_labels)
                
                val_loss += batch_loss
                val_mae += batch_mae
                val_rmse += batch_rmse
                val_r2 += batch_r2
                val_mape += batch_mape
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'MAE': f'{batch_mae:.4f}',
                    'R²': f'{batch_r2:.4f}'
                })
        
        # Average metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mae /= len(train_loader)
        val_mae /= len(val_loader)
        train_rmse /= len(train_loader)
        val_rmse /= len(val_loader)
        train_r2 /= len(train_loader)
        val_r2 /= len(val_loader)
        train_mape /= len(train_loader)
        val_mape /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        train_mapes.append(train_mape)
        val_mapes.append(val_mape)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('MAE/Train', train_mae, epoch)
            writer.add_scalar('MAE/Validation', val_mae, epoch)
            writer.add_scalar('RMSE/Train', train_rmse, epoch)
            writer.add_scalar('RMSE/Validation', val_rmse, epoch)
            writer.add_scalar('R2/Train', train_r2, epoch)
            writer.add_scalar('R2/Validation', val_r2, epoch)
            writer.add_scalar('MAPE/Train', train_mape, epoch)
            writer.add_scalar('MAPE/Validation', val_mape, epoch)
            writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Check for best model (using R² as primary criterion for hyperparameter tuning)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch_r2 = epoch
            if save_dir:
                save_checkpoint(model, optimizer, epoch, val_r2, save_dir, 'best')
        
        # Still track best loss for reference
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_loss = epoch
        
        # Track best metrics for other criteria
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch_mae = epoch
            
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch_rmse = epoch
            
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_epoch_mape = epoch
        
        # Note: Only saving best and last checkpoints, no periodic saves
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Loss': f'{train_loss:.4f}/{val_loss:.4f}',
            'MAE': f'{train_mae:.4f}/{val_mae:.4f}',
            'R²': f'{train_r2:.4f}/{val_r2:.4f}',
            'RMSE': f'{train_rmse:.4f}/{val_rmse:.4f}',
            'Best R²': f'{best_val_r2:.4f}'
        })
    
    # Save final checkpoint
    if save_dir:
        save_checkpoint(model, optimizer, epochs-1, val_loss, save_dir, 'last')
    
    if writer:
        writer.close()
    
    # Prepare best metrics summary
    best_metrics = {
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'best_val_rmse': best_val_rmse,
        'best_val_r2': best_val_r2,
        'best_val_mape': best_val_mape,
        'best_epoch_loss': best_epoch_loss,
        'best_epoch_mae': best_epoch_mae,
        'best_epoch_rmse': best_epoch_rmse,
        'best_epoch_r2': best_epoch_r2,
        'best_epoch_mape': best_epoch_mape
    }
    
    return train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s, train_mapes, val_mapes, best_metrics

def plot_losses(train_losses, val_losses, train_maes=None, val_maes=None, 
                train_rmses=None, val_rmses=None, train_r2s=None, val_r2s=None, 
                train_mapes=None, val_mapes=None):
    """Plot training and validation metrics"""
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE plot
    if train_maes is not None and val_maes is not None:
        axes[0, 1].plot(train_maes, label='Training MAE')
        axes[0, 1].plot(val_maes, label='Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # RMSE plot
    if train_rmses is not None and val_rmses is not None:
        axes[0, 2].plot(train_rmses, label='Training RMSE')
        axes[0, 2].plot(val_rmses, label='Validation RMSE')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].set_title('Root Mean Squared Error')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # R² plot
    if train_r2s is not None and val_r2s is not None:
        axes[1, 0].plot(train_r2s, label='Training R²')
        axes[1, 0].plot(val_r2s, label='Validation R²')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # MAPE plot
    if train_mapes is not None and val_mapes is not None:
        axes[1, 1].plot(train_mapes, label='Training MAPE')
        axes[1, 1].plot(val_mapes, label='Validation MAPE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].set_title('Mean Absolute Percentage Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Hide the last subplot if not needed
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    criterion = nn.MSELoss()
    
    # Evaluation progress bar
    eval_pbar = tqdm(test_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch_data, batch_labels in eval_pbar:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels)
            test_loss += loss.item()
            test_mae += torch.mean(torch.abs(outputs.squeeze() - batch_labels)).item()
            
            # Update evaluation progress bar
            eval_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{torch.mean(torch.abs(outputs.squeeze() - batch_labels)).item():.4f}'
            })
    
    return test_loss / len(test_loader), test_mae / len(test_loader)

def save_checkpoint(model, optimizer, epoch, val_loss, save_dir, name):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'model_config': {
            'input_channels': 2,
            'hidden_channels': 64,
            'kernel_size': (1, 3),
            'dropout': 0.5
        }
    }
    
    checkpoint_path = os.path.join(save_dir, f'{name}.pth')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model=None, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'val_loss': checkpoint['val_loss'],
        'model_config': checkpoint.get('model_config', {})
    }

def resume_training(model, train_loader, val_loader, checkpoint_path, 
                   epochs=250, lr=0.0001, device='cpu', log_dir=None, 
                   weight_decay=0.001, save_dir=None):
    """Resume training from checkpoint"""
    print(f"🔄 Resuming training from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(checkpoint_path, model, None, device)
    start_epoch = checkpoint_info['epoch'] + 1
    
    print(f"   Starting from epoch: {start_epoch}")
    print(f"   Previous val_loss: {checkpoint_info['val_loss']:.4f}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    load_checkpoint(checkpoint_path, None, optimizer, device)
    
    # Continue training
    return train_model(model, train_loader, val_loader, epochs, lr, device, 
                      log_dir, weight_decay, save_dir, start_epoch)

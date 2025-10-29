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
    
    # Checkpoint tracking
    best_val_loss = float('inf')
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, epochs), desc="Training", leave=True)
    
    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
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
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs.squeeze() - batch_labels)).item()
            
            # Update training progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{torch.mean(torch.abs(outputs.squeeze() - batch_labels)).item():.4f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                       leave=False, disable=len(val_loader) < 10)
        
        with torch.no_grad():
            for batch_data, batch_labels in val_pbar:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels)
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs.squeeze() - batch_labels)).item()
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{torch.mean(torch.abs(outputs.squeeze() - batch_labels)).item():.4f}'
                })
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mae /= len(train_loader)
        val_mae /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('MAE/Train', train_mae, epoch)
            writer.add_scalar('MAE/Validation', val_mae, epoch)
            writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Check for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if save_dir:
                save_checkpoint(model, optimizer, epoch, val_loss, save_dir, 'best')
        
        # Note: Only saving best and last checkpoints, no periodic saves
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Train MAE': f'{train_mae:.4f}',
            'Val MAE': f'{val_mae:.4f}',
            'Best Val': f'{best_val_loss:.4f}'
        })
    
    # Save final checkpoint
    if save_dir:
        save_checkpoint(model, optimizer, epochs-1, val_loss, save_dir, 'last')
    
    if writer:
        writer.close()
    
    return train_losses, val_losses, train_maes, val_maes

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
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

"""
Hyperparameter tuning module using Optuna
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import os
import json
import ast
from typing import Dict, Any, Tuple
from datetime import datetime

from .data import load_data, get_dataloaders
from .model import ConvLSTMRegressor
from .trainer import train_model
from .metrics import calculate_r2_score


class OptunaTuner:
    """Hyperparameter tuner using Optuna"""
    
    def __init__(self, config_path: str, data_path: str, device: str = 'cpu'):
        self.config_path = config_path
        self.data_path = data_path
        self.device = device
        self.tune_config = self._load_tune_config()
        self.study = None
        self.best_params = None
        self.best_score = float('inf')
        
    def _load_tune_config(self) -> Dict[str, Any]:
        """Load tuning configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_objective(self, X_train, X_test, y_train, y_test):
        """Create objective function for Optuna"""
        
        def objective(trial):
            # Sample hyperparameters from the search space
            params = self._sample_params(trial)
            
            # Create model with sampled parameters
            model = ConvLSTMRegressor(
                input_channels=2,
                hidden_channels=params['hidden_channels'],
                kernel_size=params['kernel_size'],
                dropout=params['dropout']
            )
            
            # Create dataloaders
            train_loader, test_loader = get_dataloaders(
                X_train, X_test, y_train, y_test,
                batch_size=params['batch_size']
            )
            
            # Train model with early stopping based on validation R² score
            try:
                train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s, train_mapes, val_mapes, best_metrics = train_model(
                    model, train_loader, test_loader,
                    epochs=params['epochs'],
                    lr=params['learning_rate'],
                    device=self.device,
                    log_dir=None,  # No TensorBoard during tuning
                    weight_decay=params['weight_decay'],
                    save_dir=None  # No checkpointing during tuning
                )
                
                # Return best validation R² score as the metric to maximize
                final_val_r2 = max(val_r2s)
                
                # Report intermediate values for pruning
                for epoch, (train_loss, val_loss, train_mae, val_mae, train_r2, val_r2) in enumerate(
                    zip(train_losses, val_losses, train_maes, val_maes, train_r2s, val_r2s)
                ):
                    trial.report(val_r2, epoch)  # Report R² directly
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return final_val_r2  # Return R² directly for maximization
                
            except Exception as e:
                print(f"Trial failed with error: {e}")
                return -float('inf')  # Very bad R² score for failed trials
        
        return objective
    
    def _sample_params(self, trial) -> Dict[str, Any]:
        """Sample hyperparameters from the search space"""
        params = {}
        
        # Model parameters (tuned)
        model_config = self.tune_config['model']
        params['hidden_channels'] = trial.suggest_categorical(
            'hidden_channels', model_config['hidden_channels']['choices']
        )
        kernel_choice = trial.suggest_categorical(
            'kernel_size', model_config['kernel_size']['choices']
        )
        # Convert to tuple for the model (handle both list and tuple cases)
        if isinstance(kernel_choice, (list, tuple)):
            params['kernel_size'] = tuple(kernel_choice)
        else:
            # If it's a string representation, try to parse it safely
            try:
                params['kernel_size'] = ast.literal_eval(kernel_choice)
            except:
                # Fallback: assume it's a list-like string
                params['kernel_size'] = (1, 3)  # Default fallback
        params['dropout'] = trial.suggest_float(
            'dropout', 
            model_config['dropout']['low'], 
            model_config['dropout']['high']
        )
        
        # Training parameters (fixed)
        training_config = self.tune_config['training']
        if isinstance(training_config['learning_rate'], dict):
            params['learning_rate'] = trial.suggest_float(
                'learning_rate',
                training_config['learning_rate']['low'],
                training_config['learning_rate']['high'],
                log=True
            )
        else:
            params['learning_rate'] = training_config['learning_rate']
            
        if isinstance(training_config['weight_decay'], dict):
            params['weight_decay'] = trial.suggest_float(
                'weight_decay',
                training_config['weight_decay']['low'],
                training_config['weight_decay']['high'],
                log=True
            )
        else:
            params['weight_decay'] = training_config['weight_decay']
            
        if isinstance(training_config['batch_size'], dict):
            params['batch_size'] = trial.suggest_categorical(
                'batch_size', training_config['batch_size']['choices']
            )
        else:
            params['batch_size'] = training_config['batch_size']
            
        # Epochs is fixed (not tunable)
        params['epochs'] = training_config['epochs']
        
        # Data parameters (fixed)
        data_config = self.tune_config['data']
        if isinstance(data_config['test_size'], dict):
            params['test_size'] = trial.suggest_float(
                'test_size',
                data_config['test_size']['low'],
                data_config['test_size']['high']
            )
        else:
            params['test_size'] = data_config['test_size']
        
        return params
    
    def tune(self, save_dir: str = None) -> Dict[str, Any]:
        """Run hyperparameter tuning"""
        print("🔍 Starting hyperparameter tuning with Optuna...")
        
        # Load data
        print("📊 Loading data...")
        test_size = self.tune_config['data']['test_size']
        X_train, X_test, y_train, y_test, scaler = load_data(
            csv_path=self.data_path,
            test_size=test_size
        )
        
        # Create study
        study_name = f"gripper_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_url = None
        
        if save_dir:
            # Ensure directory exists before creating database
            os.makedirs(save_dir, exist_ok=True)
            storage_url = f"sqlite:///{save_dir}/optuna_study.db"
        
        self.study = optuna.create_study(
            direction=self.tune_config['tuning']['direction'],
            pruner=optuna.pruners.MedianPruner() if self.tune_config['tuning']['pruner'] == 'median' else None,
            sampler=optuna.samplers.TPESampler() if self.tune_config['tuning']['sampler'] == 'tpe' else None,
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True
        )
        
        # Create objective function
        objective = self._create_objective(X_train, X_test, y_train, y_test)
        
        # Run optimization
        print(f"🚀 Running {self.tune_config['tuning']['n_trials']} trials...")
        self.study.optimize(
            objective,
            n_trials=self.tune_config['tuning']['n_trials'],
            timeout=self.tune_config['tuning']['timeout']
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"✅ Tuning completed!")
        print(f"🎯 Best validation R²: {self.best_score:.4f}")
        print(f"🏆 Best parameters: {self.best_params}")
        
        # Save results
        if save_dir:
            self._save_results(save_dir)
        
        return self.best_params
    
    def _save_results(self, save_dir: str):
        """Save tuning results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best parameters as JSON
        best_params_file = os.path.join(save_dir, 'best_params.json')
        with open(best_params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save best parameters as train config YAML
        self._save_best_config(save_dir)
        
        # Save study summary
        study_summary = {
            'best_value': self.best_score,
            'best_params': self.best_params,
            'n_trials': len(self.study.trials),
            'best_trial_number': self.study.best_trial.number,
            'tuning_config': self.tune_config
        }
        
        summary_file = os.path.join(save_dir, 'tuning_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(study_summary, f, indent=2)
        
        print(f"💾 Results saved to: {save_dir}")
    
    def _save_best_config(self, save_dir: str):
        """Save best parameters as train config YAML"""
        # Create train config with best parameters
        train_config = {
            'hidden_channels': self.best_params['hidden_channels'],
            'kernel_size': self.best_params['kernel_size'],
            'dropout': self.best_params['dropout'],
            'epochs': self.tune_config['training']['epochs'],  # Use fixed epochs value from config
            'batch_size': self.best_params['batch_size'],
            'learning_rate': self.best_params['learning_rate'],
            'weight_decay': self.best_params['weight_decay'],
            'test_size': self.tune_config['data']['test_size']  # Use fixed test_size from config
        }
        
        # Save as YAML file
        config_file = os.path.join(save_dir, 'best_config.yaml')
        with open(config_file, 'w') as f:
            f.write("# ConvLSTM Training Configuration\n")
            f.write("# Best hyperparameters from Optuna tuning\n\n")
            f.write("# Model Architecture\n")
            f.write(f"hidden_channels: {train_config['hidden_channels']}\n")
            f.write(f"kernel_size: {train_config['kernel_size']}\n")
            f.write(f"dropout: {train_config['dropout']}\n\n")
            f.write("# Training Hyperparameters\n")
            f.write(f"epochs: {train_config['epochs']}\n")
            f.write(f"batch_size: {train_config['batch_size']}\n")
            f.write(f"learning_rate: {train_config['learning_rate']}\n")
            f.write(f"weight_decay: {train_config['weight_decay']}\n\n")
            f.write("# Data Split\n")
            f.write(f"test_size: {train_config['test_size']}\n")
        
        print(f"📄 Best config saved as: {config_file}")
    
    def get_best_model(self) -> ConvLSTMRegressor:
        """Get model with best parameters"""
        if self.best_params is None:
            raise ValueError("No tuning results available. Run tune() first.")
        
        return ConvLSTMRegressor(
            input_channels=2,
            hidden_channels=self.best_params['hidden_channels'],
            kernel_size=self.best_params['kernel_size'],
            dropout=self.best_params['dropout']
        )
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        try:
            import matplotlib.pyplot as plt
            import optuna.visualization as vis
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Optimization history
            vis.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
            
            # Parameter importance
            vis.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
            
            # Parallel coordinate plot
            vis.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
            
            # Slice plot
            vis.matplotlib.plot_slice(self.study, ax=axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Optimization plots saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️  Matplotlib not available for plotting. Install with: pip install matplotlib")


def create_tune_config_template(output_path: str):
    """Create a template tuning configuration file"""
    template_content = """# Optuna Hyperparameter Tuning Space Configuration
# Define the search space for hyperparameter optimization

# Model Architecture Parameters
model:
  hidden_channels:
    type: "categorical"
    choices: [32, 64, 128, 256]
  
  kernel_size:
    type: "categorical" 
    choices: [(1, 3), (3, 3), (5, 5)]
  
  dropout:
    type: "uniform"
    low: 0.1
    high: 0.7

# Training Hyperparameters
training:
  learning_rate:
    type: "loguniform"
    low: 0.00001
    high: 0.01
  
  weight_decay:
    type: "loguniform"
    low: 0.00001
    high: 0.1
  
  batch_size:
    type: "categorical"
    choices: [4, 8, 16, 32]
  
  epochs: 50  # Fixed epochs for tuning (not tunable)

# Data Parameters (fixed - not tuned)
data:
  test_size: 0.3

# Tuning Configuration
tuning:
  n_trials: 50
  timeout: 3600  # 1 hour timeout
  direction: "minimize"  # minimize MAE
  metric: "val_mae"  # target metric for optimization
  pruner: "median"  # pruning strategy
  sampler: "tpe"  # sampling strategy
"""
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    print(f"📝 Tuning config template created: {output_path}")

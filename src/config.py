"""
Simple hyperparameter configuration management for ConvLSTM training
"""
import yaml
import argparse
from typing import Dict, Any, Optional


def load_hyperparams(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load hyperparameters from YAML file or use defaults"""
    
    # Default hyperparameters
    defaults = {
        'hidden_channels': 64,
        'kernel_size': [1, 3],
        'dropout': 0.5,
        'epochs': 250,
        'batch_size': 8,
        'learning_rate': 0.0001,
        'weight_decay': 0.001,
        'test_size': 0.3
    }
    
    if config_path is None:
        print("🔄 Using default hyperparameters")
        return defaults
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults (YAML overrides defaults)
        hyperparams = defaults.copy()
        hyperparams.update(config)
        
        print(f"📄 Loaded hyperparameters from: {config_path}")
        return hyperparams
        
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("🔄 Using default hyperparameters")
        return defaults
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config: {e}")
        print("🔄 Using default hyperparameters")
        return defaults


def update_hyperparams_from_args(hyperparams: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update hyperparameters with command line arguments (override YAML)"""
    updated = hyperparams.copy()
    
    # Model hyperparameters
    if hasattr(args, 'hidden_channels') and args.hidden_channels is not None:
        updated['hidden_channels'] = args.hidden_channels
    if hasattr(args, 'kernel_size') and args.kernel_size:
        updated['kernel_size'] = args.kernel_size
    if hasattr(args, 'dropout') and args.dropout is not None:
        updated['dropout'] = args.dropout
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        updated['weight_decay'] = args.weight_decay
    
    # Training hyperparameters
    if hasattr(args, 'epochs') and args.epochs is not None:
        updated['epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        updated['batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        updated['learning_rate'] = args.lr
    if hasattr(args, 'test_size') and args.test_size is not None:
        updated['test_size'] = args.test_size
    
    return updated


def print_hyperparams(hyperparams: Dict[str, Any]) -> None:
    """Print current hyperparameters"""
    print("🔧 Model Hyperparameters:")
    print("=" * 40)
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("=" * 40)

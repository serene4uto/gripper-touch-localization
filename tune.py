#!/usr/bin/env python3
"""
Hyperparameter Tuning Script using Optuna
"""
import argparse
import os
import sys
from datetime import datetime

from src.tuner import OptunaTuner, create_tune_config_template


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    
    # Required arguments
    parser.add_argument('--data-path', type=str, required=False,
                        help='Path to training data CSV file')
    
    # Optional arguments
    parser.add_argument('--tune-config', type=str, 
                        default='configs/tune/default_tune_cfg.yaml',
                        help='Path to tuning configuration file')
    
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: exp_YYYYMMDD_HHMMSS)')
    
    parser.add_argument('--tune-dir', type=str, default='tune_results',
                        help='Base directory for tuning results (default: tune_results)')
    
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    
    parser.add_argument('--create-template', action='store_true',
                        help='Create a template tuning configuration file')
    
    parser.add_argument('--plot', action='store_true',
                        help='Generate optimization plots after tuning')
    
    return parser.parse_args()


def main():
    """Main tuning function"""
    args = parse_args()
    
    # Create template if requested
    if args.create_template:
        template_path = 'configs/tune/template.yaml'
        create_tune_config_template(template_path)
        print(f"✅ Template created: {template_path}")
        return
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Check if tuning config exists
    if not os.path.exists(args.tune_config):
        print(f"❌ Tuning config not found: {args.tune_config}")
        print("💡 Create one with: python tune.py --create-template")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🔧 Using device: {device}")
    
    # Set up experiment name and directory
    if args.exp_name is None:
        exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        exp_name = args.exp_name
    
    output_dir = os.path.join(args.tune_dir, exp_name)
    
    # Create tuner
    tuner = OptunaTuner(
        config_path=args.tune_config,
        data_path=args.data_path,
        device=device
    )
    
    # Run tuning
    print(f"🚀 Starting hyperparameter tuning...")
    print(f"📁 Results will be saved to: {output_dir}")
    
    best_params = tuner.tune(save_dir=output_dir)
    
    # Generate plots if requested
    if args.plot:
        plot_path = os.path.join(output_dir, 'optimization_plots.png')
        try:
            tuner.plot_optimization_history(plot_path)
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 TUNING COMPLETED")
    print("="*60)
    print(f"📊 Best validation MAE: {tuner.best_score:.4f}")
    print(f"🏆 Best parameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔬 Experiment: {exp_name}")
    print("="*60)


if __name__ == "__main__":
    main()

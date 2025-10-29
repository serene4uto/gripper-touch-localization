# Gripper Touch Localization

PyTorch implementation of ConvLSTM for contact area prediction in robotic gripper systems.

## Features

- **ConvLSTM Model**: Deep learning model for sequential contact area prediction
- **Hyperparameter Tuning**: Automated optimization using Optuna
- **Comprehensive Metrics**: MAE, MSE, RMSE, R², MAPE evaluation
- **Best Model Tracking**: Saves models based on R² score optimization
- **Visualization**: Training plots and prediction analysis
- **Easy Evaluation**: Simple evaluation pipeline for trained models

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py --data-path dataset/SizeTrain.csv --exp-name my_experiment
```

### 3. Evaluate Model
```bash
python eval.py --model-path runs/my_experiment/model.pth --plot-results --save-predictions
```

### 4. Hyperparameter Tuning
```bash
python tune.py --data-path dataset/SizeTrain.csv --exp-name tuning_run
```

## Usage

### Training
```bash
# Basic training
python train.py --data-path dataset/SizeTrain.csv

# With custom experiment name
python train.py --data-path dataset/SizeTrain.csv --exp-name my_run

# Resume from checkpoint
python train.py --resume runs/exp_123/checkpoint/best.pth
```

### Evaluation
```bash
# Basic evaluation
python eval.py --model-path runs/exp_123/model.pth

# With plots and predictions
python eval.py --model-path runs/exp_123/model.pth --plot-results --save-predictions

# Custom output directory
python eval.py --model-path runs/exp_123/model.pth --output-dir my_results
```

### Hyperparameter Tuning
```bash
# Run tuning
python tune.py --data-path dataset/SizeTrain.csv --exp-name tuning_run

# Create tuning config template
python tune.py --create-template
```

## Project Structure

```
├── train.py              # Training script
├── eval.py               # Evaluation script  
├── tune.py               # Hyperparameter tuning script
├── src/
│   ├── model.py          # ConvLSTM model definition
│   ├── trainer.py        # Training logic and metrics
│   ├── data.py           # Data loading and preprocessing
│   ├── config.py         # Configuration management
│   └── metrics.py        # Evaluation metrics
├── configs/
│   ├── train/            # Training configurations
│   └── tune/             # Tuning configurations
├── runs/                 # Training outputs
├── tune_results/         # Tuning outputs
└── dataset/              # Data files
```

## Configuration

- **Training**: Edit `configs/train/default_cfg.yaml`
- **Tuning**: Edit `configs/tune/default_tune_cfg.yaml`
- **Model**: ConvLSTM with configurable hidden channels, kernel size, dropout

## Outputs

### Training
- `runs/exp_YYYYMMDD_HHMMSS/`
  - `model.pth` - Final trained model
  - `checkpoint/best.pth` - Best model (highest R²)
  - `best_results.txt` - Comprehensive results summary
  - `config.yaml` - Training configuration used

### Evaluation
- `eval_results/`
  - `evaluation_report.txt` - Detailed metrics
  - `predictions.csv` - Raw predictions (optional)
  - `evaluation_plots.png` - Visualization plots (optional)

### Tuning
- `tune_results/exp_YYYYMMDD_HHMMSS/`
  - `best_config.yaml` - Best hyperparameters found
  - `optuna_study.db` - Optuna study database
  - Individual trial results

## Metrics

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination (optimization target)
- **MAPE**: Mean Absolute Percentage Error

## Requirements

- Python 3.7+
- PyTorch
- Optuna (for hyperparameter tuning)
- scikit-learn
- matplotlib
- pandas
- tqdm
- tensorboard

## License

MIT License
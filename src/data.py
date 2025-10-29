import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class GripperDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_data(csv_path="dataset/SizeTrain.csv", test_size=0.3):
    """Load and preprocess data"""
    # Load data
    data = pd.read_csv(csv_path)
    features = ["S1", "S2", "S3", "SIZE", "STD_POINT", "FORCE"]
    target = "AREA_AVG"
    
    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[features])
    y = data[target].values
    
    # Reshape for ConvLSTM: (samples, timesteps, channels, height, width)
    # Original: 6 features -> reshape to (1, 2, 1, 3) = 6 elements
    X = X.reshape(-1, 1, 2, 1, 3)  # (N, T=1, C=2, H=1, W=3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def get_dataloaders(X_train, X_test, y_train, y_test, batch_size=8):
    """Create PyTorch dataloaders"""
    train_dataset = GripperDataset(X_train, y_train)
    test_dataset = GripperDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

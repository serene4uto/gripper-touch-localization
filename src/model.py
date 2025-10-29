import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM2D(nn.Module):
    """Simple ConvLSTM2D implementation"""
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM2D, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        # Convolution layers for gates
        self.conv_i = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_f = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_g = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_o = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        
    def forward(self, x, hidden=None):
        batch_size, seq_len, channels, height, width = x.size()
        
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h, c = hidden
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            combined = torch.cat([x_t, h], dim=1)
            
            # Gates
            i = torch.sigmoid(self.conv_i(combined))
            f = torch.sigmoid(self.conv_f(combined))
            g = torch.tanh(self.conv_g(combined))
            o = torch.sigmoid(self.conv_o(combined))
            
            # Cell state and hidden state
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            outputs.append(h)
        
        return torch.stack(outputs, dim=1), (h, c)

class ConvLSTMRegressor(nn.Module):
    """ConvLSTM-based regression model for tactile sensing"""
    def __init__(self, input_channels=2, hidden_channels=64, kernel_size=(1, 3), dropout=0.5):
        super(ConvLSTMRegressor, self).__init__()
        
        # ConvLSTM2D layer
        self.convlstm = ConvLSTM2D(input_channels, hidden_channels, kernel_size)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        # Calculate the correct input size: hidden_channels * height * width
        self.fc1 = nn.Linear(hidden_channels * 1 * 3, 32)  # 64 * 1 * 3 = 192
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, timesteps, channels, height, width)
        # ConvLSTM2D
        lstm_out, _ = self.convlstm(x)
        
        # Take the last timestep output
        last_output = lstm_out[:, -1, :, :, :]  # (batch, hidden_channels, height, width)
        
        # Flatten and pass through FC layers
        x = self.flatten(last_output)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation for regression
        
        return x

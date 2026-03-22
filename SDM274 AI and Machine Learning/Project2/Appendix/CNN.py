import torch.nn as nn

class CNNTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [batch, time, feature]
        x = x.permute(0, 2, 1)   # → [batch, feature, time]
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        x = self.global_pool(x).squeeze(-1)  # → [batch, 64]
        out = self.fc(x)                     # → [batch, output_dim]
        
        return out

import torch.nn as nn

class LSTMTimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, time, feature]
        
        # LSTM 输出：
        # out: [batch, time, hidden_dim]
        out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的隐藏状态
        last_hidden = out[:, -1, :]   # [batch, hidden_dim]
        
        output = self.fc(last_hidden)  # [batch, output_dim]
        return output

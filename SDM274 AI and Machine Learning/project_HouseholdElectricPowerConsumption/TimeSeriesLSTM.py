import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm_layer = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.regularization = nn.Dropout(0.2)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences):
        lstm_out, _ = self.lstm_layer(sequences)
        last_timestep = lstm_out[:, -1, :]
        regularized = self.regularization(last_timestep)
        predictions = self.output_layer(regularized)
        return predictions
import torch.nn as nn
class TimeSeriesCNN(nn.Module):
    def __init__(self, feature_channels, sequence_length, prediction_dim):
        super(TimeSeriesCNN, self).__init__()
        self.convolution = nn.Conv1d(in_channels=feature_channels, out_channels=64, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.regularization = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64 * sequence_length, prediction_dim)

    def forward(self, inputs):
        features = self.convolution(inputs)
        activated = self.activation(features)
        regularized = self.regularization(activated)
        flattened = regularized.view(regularized.size(0), -1)
        predictions = self.output_layer(flattened)
        return predictions
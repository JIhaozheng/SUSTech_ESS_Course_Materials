import torch.nn as nn
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, attention_heads, encoder_layers, dropout_rate=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=encoder_layers
        )
        
        self.regularization = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences):
        embedded = self.feature_projection(sequences)
        embedded = embedded.permute(1, 0, 2)
        encoded = self.transformer_encoder(embedded)
        last_timestep = encoded[-1, :, :]
        regularized = self.regularization(last_timestep)
        predictions = self.output_layer(regularized)
        
        return predictions
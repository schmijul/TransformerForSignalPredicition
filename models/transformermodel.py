import math
import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, input_shape, output_shape, num_layers=6, d_model=64, num_heads=4, dff=128, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(input_shape[1], d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dff, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Add more linear layers
        self.linear1 = nn.Linear(d_model, dff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dff, output_shape)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)

        # Pass through the additional linear layers and activation
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x

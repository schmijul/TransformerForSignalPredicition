import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        x = x[:, -1, :]  # letztes Element der Sequenz auswählen
        x = self.output(x)
        return x

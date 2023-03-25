import math
import torch
import torch.nn as nn



class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_embedding = torch.zeros(1, max_len, d_model)
        pos_embedding[0, :, 0::2] = torch.sin(pos * div)
        pos_embedding[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x
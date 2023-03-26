import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer

class AutoregressiveTransformer(LightningModule):
    def __init__(self, d_model, nhead, num_layers, input_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.linear = nn.Linear(input_size, d_model)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(1)

        # Create the target tensor by shifting the input tensor by one position
        tgt = x.roll(shifts=-1, dims=1)
        tgt[:, -1, :] = 0  # Zero out the last position of the target tensor

        # Pass both source and target tensors to the transformer
        output = self.transformer(x, tgt)
        return output[:, 0, :]


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss)
import os
import numpy as np
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
        """
        This function implements the forward pass of the model.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        """
        x = self.linear(x)
        x = x.unsqueeze(1)

        # Create the target tensor by shifting the input tensor by one position
        tgt = x.roll(shifts=-1, dims=1)
        tgt[:, -1, :] = 0  # Zero out the last position of the target tensor

        # Pass both source and target tensors to the transformer
        output = self.transformer(x, tgt)
        return output[:, 0, :]


    def configure_optimizers(self):
        """
        This function configures the optimizer.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        this function implements the training step.
        :param batch: Batch of data
        :param batch_idx: Batch index 

        :return: Loss (torch.Tensor)
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        This function implements the validation step.
        :param batch: Batch of data
        :param batch_idx: Batch index
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """
        This function implements the test step.
        :param batch: Batch of data
        :param batch_idx: Batch index  
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss)
        
        
    def predict_test_set(self, test_dataloader):
        """
        This function predicts the test set.
        :param test_dataloader: Dataloader for the test set (torch.utils.data.DataLoader)
        : return: Predictions   (numpy.ndarray)
        """
        self.eval()  # Set the model to evaluation mode
        predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                x, y = batch
                y_hat = self(x)
                predictions.append(y_hat.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        horizon = predictions.shape[1] - 1
        np.save(f"prediction_ART_SNR10_{horizon+1}ms.npy", predictions)

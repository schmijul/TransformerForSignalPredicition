import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchviz import make_dot


class AutoregressiveTransformer(pl.LightningModule):
    """
    This class implements an autoregressive transformer model. 
    The model is based on the transformer architecture 
    from the paper "Attention is all you need" (https://arxiv.org/abs/1706.03762).
    The model is trained to predict the next value in a 
    time series given the previous values in the time series.

    """
    def __init__(self, d_model: int, nhead: int, num_layers: int, input_size: int):
        """
        :param d_model: Size of the input to the transformer
        :param nhead: Number of heads in the multi-head attention layer
        :param num_layers: Number of transformer layers
        :param input_size: Size of the input to the model        
        """
        super().__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.linear = nn.Linear(input_size, d_model)

    def forward(self, x: torch.Tensor):
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

    def training_step(self, batch: tuple, batch_idx: int):
        """
        this function implements the training step.
        :param batch: Batch of data
        :return: Dictionary with "loss" key (torch.Tensor)
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        This function implements the validation step.
        :param batch: Batch of data

        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch: tuple, batch_idx: int):
        """
        This function implements the test step.
        :param batch: Batch of data
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss)

    def predict_test_set(self, test_dataloader: DataLoader):
        """
        This function predicts the test set.
        :param test_dataloader: Dataloader for the test set (torch.utils.data.DataLoader)
        :return: Predictions (numpy.ndarray)
        """
        self.eval()  # Set the model to evaluation mode
        predictions = []

        with torch.no_grad():
            for batch in test_dataloader:
                x = batch[0]
                y_hat = self(x)
                predictions.append(y_hat.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        horizon = predictions.shape[1] - 1
        np.save(f"prediction_ART_SNR10_{horizon+1}ms.npy", predictions)
        return predictions

    def plot(self, x):
        """
        This function generates a plot of the computational graph of the model.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        """
        y = self(x)
        make_dot(y, params=dict(self.named_parameters())).render("autoregressive_transformer", format="png")
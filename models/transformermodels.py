

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchviz import make_dot
class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self, d_model, nhead, num_layers, seq_len=25):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.input_linear = nn.Linear(1, d_model)  # Add this line
        self.transformer_layer = nn.Transformer(d_model, nhead, num_layers)
        self.linear = nn.Linear(d_model, 1)
        
    def create_positional_encoding(self, batch_size=1):
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model)).unsqueeze(0)
        positional_encoding = torch.zeros(self.seq_len, self.d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        if self.d_model > 1:
            positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(1).expand(-1, batch_size, -1)  # Add batch dimension and expand
        return positional_encoding





    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # Add channel dimension: (batch_size, seq_len) -> (batch_size, seq_len, 1)
        x = x.permute(1, 0, 2)  # Permute dimensions: (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        pos_enc = self.create_positional_encoding(batch_size=batch_size)  # Pass batch_size argument here
        x = x + pos_enc
        x = self.transformer_layer(x, x)
        x = self.linear(x[-1])
        x = x.permute(1, 0)  # Permute dimensions: (seq_len, batch_size) -> (batch_size, seq_len)
        return x.squeeze()





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

    def predict_step(self, batch: tuple, batch_idx: int):
        """
        This function implements the predict step.
        :param batch: Batch of data
        :return: Predictions (torch.Tensor)
        """
        x, _ = batch
        y_hat = self(x)
        return y_hat
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
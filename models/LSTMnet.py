import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchviz import make_dot
import numpy as np
import pytorch_lightning as pl

class LSTMnet(nn.Module):
    def __init__(self, num_units):
        super(LSTMnet, self).__init__()
        self.num_units = num_units

        self.lstm_1 = nn.LSTM(1, self.num_units, batch_first=True)
        self.drop_1 = nn.Dropout(0.5)
        self.lstm_2 = nn.LSTM(self.num_units, self.num_units, batch_first=True)
        self.dense_1 = nn.Linear(self.num_units, 400)
        self.drop_2 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(400, 1300)
        self.flat = nn.Flatten()
        self.dense_3 = nn.Linear(1300, 200)
        self.pred_layer = nn.Linear(200, 1)

    def forward(self, inputs):
        x, _ = self.lstm_1(inputs)
        x = self.drop_1(x)
        x, _ = self.lstm_2(x)
        x = self.dense_1(x[:, -1, :])
        x = self.drop_2(x)
        x = F.relu(x)
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.flat(x)
        x = self.dense_3(x)
        x = F.relu(x)
        x = self.pred_layer(x)
        return x

    def configure_optimizers(self):
        """
        This function configures the optimizer.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch: tuple):
        """
        This function implements the training step.
        :param batch: Batch of data
        :return: Dictionary with "loss" key (torch.Tensor)
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch: tuple):
        """
        This function implements the validation step.
        :param batch: Batch of data

        """
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch: tuple):
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
        np.save(f"prediction_LSTM_SNR10_{horizon+1}ms.npy", predictions)
        return predictions

    def plot(self, x):
        """
        This function generates a plot of the computational graph of the model.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        """
        y = self(x)
        make_dot(y, params=dict(self.named_parameters())).render("autoregressive_transformer", format="png")
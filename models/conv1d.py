import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchviz import make_dot
import numpy as np
import pytorch_lightning as pl

class CONV1D(nn.Module):
    def __init__(self, num_filter, kernel_size, num_neurons):
        super(CONV1D, self).__init__()
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        self.num_neurons = num_neurons
        
        self.conv_1 = nn.Conv1d(1, self.num_filter, kernel_size=self.kernel_size)
        self.flat_1 = nn.Flatten()
        self.dense_1 = nn.Linear(self.num_filter * (100 - self.kernel_size + 1), self.num_neurons)
        self.dense_2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.dense_3 = nn.Linear(self.num_neurons, 200)
        self.pred_layer = nn.Linear(200, 1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)
        x = self.flat_1(x)
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dense_2(x)
        x = F.relu(x)
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
        this function implements the training step.
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
        np.save(f"prediction_ART_SNR10_{horizon+1}ms.npy", predictions)
        return predictions

    def plot(self, x):
        """
        This function generates a plot of the computational graph of the model.
        :param x: Input tensor of shape (batch_size, seq_len, input_size)
        """
        y = self(x)
        make_dot(y, params=dict(self.named_parameters())).render("autoregressive_transformer", format="png")
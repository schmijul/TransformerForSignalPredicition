import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from models.transformermodels import AutoregressiveTransformer
from timeseries_dataset import TimeSeriesDataset


def load_dataset(path="data/", noise="snr10", horizon=10):
    train_data = {
        "x": np.load(path + f"train_data/x_train_{noise}.npy"),
        "y": np.load(path + f"train_data/y_train_{noise}.npy")[:, horizon]
    }

    val_data = {
        "x": np.load(path + f"val_data/x_val_{noise}.npy"),
        "y": np.load(path + f"val_data/y_val_{noise}.npy")[:, horizon]
    }
    test_data = {
        "x": np.load(path + f"test_data/x_test_{noise}.npy"),
        "y": np.load(path + f"test_data/y_test_{noise}.npy")[:, horizon]
    }

    return train_data, val_data, test_data

if __name__ == "__main__":
    train_data, val_data, test_data = load_dataset()

    train_dataset = TimeSeriesDataset(train_data["x"], train_data["y"])
    val_dataset = TimeSeriesDataset(val_data["x"], val_data["y"])
    test_dataset = TimeSeriesDataset(test_data["x"], test_data["y"])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model = AutoregressiveTransformer(d_model=64, nhead=4, num_layers=3, input_size=25)
    trainer = Trainer(max_epochs=10)  # Use 'gpus=1' to run on a GPU, if available
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(test_dataloaders=test_dataloader)
    model.predict_test_set(test_dataloader)
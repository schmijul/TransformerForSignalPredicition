import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from models.transformermodels import AutoregressiveTransformer
from timeseries_dataset import TimeSeriesDataset


def load_dataset(path="data/", noise="snr10", horizon=3):
    """
    This function loads the dataset from the specified path.
    :param path: Path to the dataset
    :param noise: SNR in dB
    :param horizon: Prediction horizon ( ms)
    
    :return: train_data, val_data, test_data (dict)
    """
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
    
    BATCH_SIZE = 4*64 # Batch size
    NUM_LAYERS = 2 # Number of transformer layers
    N_HEAD = 2 # Number of heads in the multi-head attention layer
    MAX_EPOCHS = 100 # Maximum number of epochs
    PATIENCE = 10 # Early stopping patience if val loss does not improve after PATIENCE epochs stop Training
    
    # Load the dataset

    train_data, val_data, test_data = load_dataset() 
    
    
    train_dataset = TimeSeriesDataset(train_data["x"], train_data["y"])
    val_dataset = TimeSeriesDataset(val_data["x"], val_data["y"])
    test_dataset = TimeSeriesDataset(test_data["x"], test_data["y"])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    # Initialize the model
    model = AutoregressiveTransformer(d_model=BATCH_SIZE, nhead=N_HEAD, num_layers=NUM_LAYERS, input_size=train_data["x"].shape[1])

    # Initialize the trainer
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    
    # Early stopping    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=PATIENCE)



    # Initialize the trainer
    trainer = Trainer(max_epochs=MAX_EPOCHS,
                      callbacks=[early_stop_callback],
                      logger=logger)
    # Training
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Testing
    trainer.test(dataloaders=test_dataloader)
    
    # Predicting
    model.predict_test_set(test_dataloader)

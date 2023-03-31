import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# Own Files
from models.transformermodels import TimeSeriesTransformer
from timeseries_dataset import TimeSeriesDataset


def load_dataset(path="data/", noise="snr10", horizon=3):
    """
    This function loads the dataset from the specified path.
    :param path: Path to the dataset
    :param noise: SNR in dB
    :param horizon: Prediction horizon (ms)
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

def main():
     # Load the dataset
    train_data, val_data, test_data = load_dataset()

    train_dataset = TimeSeriesDataset(train_data["x"], train_data["y"])
    val_dataset = TimeSeriesDataset(val_data["x"], val_data["y"])
    test_dataset = TimeSeriesDataset(test_data["x"], test_data["y"])

    train_dataloader = DataLoader(train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True, 
                                drop_last=True)
    
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                drop_last=True)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 drop_last=True)
    
    # Initialize the model
    model = TimeSeriesTransformer(
        d_model=64, 
        nhead=N_HEAD, 
        num_layers=NUM_LAYERS, 
        seq_len=train_data["x"].shape[1]
    )
   
    # Initialize PyTorch Lightning's Trainer
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')
    logger = TensorBoardLogger("tb_logs", name="TimeSeriesTransformer")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                         callbacks=[early_stopping], 
                         logger=logger)

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # save the model
    torch.save(model.state_dict(), MODEL_SAVE_NAME)
    

    # Evaluate the model on the test set
    trainer.test(test_dataloaders=test_dataloader)
    
    predictions = trainer.predict(model, test_dataloader)
    np.save(PREDICTION_FILE_NAME, predictions)
if __name__ == "__main__":
    BATCH_SIZE = 4 * 64  # Batch size
    NUM_LAYERS = 2  # Number of transformer layers
    N_HEAD = 1  # Number of heads in the multi-head attention layer
    MAX_EPOCHS = 100  # Maximum number of epochs
    PATIENCE = 10  # if val loss does not improve after PATIENCE epochs stop Training
    
    PREDICTION_FILE_NAME = "predictionsTST.npy"
    MODEL_SAVE_NAME = "TimeSeriesTransformer.pt"

    main()

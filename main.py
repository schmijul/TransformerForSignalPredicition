import os
import math
import numpy as np
import torch
import torch.nn as nn

from models.transformermodel import TransformerModel
from models.positional_encoding import PositionalEncoding


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

def main():
    train_data, val_data, test_data = load_dataset()

    for key in train_data.keys():
        if key == "x":
            train_data[key] = train_data[key].reshape(train_data[key].shape[0], train_data[key].shape[1], 1)
            val_data[key] = val_data[key].reshape(val_data[key].shape[0], val_data[key].shape[1], 1)
            test_data[key] = test_data[key].reshape(test_data[key].shape[0], test_data[key].shape[1], 1)
        elif key == "y":
            train_data[key] = train_data[key].reshape(train_data[key].shape[0], 1, 1)
            val_data[key] = val_data[key].reshape(val_data[key].shape[0], 1, 1)
            test_data[key] = test_data[key].reshape(test_data[key].shape[0], 1, 1)

    train_x = torch.tensor(train_data["x"], dtype=torch.float32)
    train_y = torch.tensor(train_data["y"], dtype=torch.float32)
    val_x = torch.tensor(val_data["x"], dtype=torch.float32)
    val_y = torch.tensor(val_data["y"], dtype=torch.float32)
    test_x = torch.tensor(test_data["x"], dtype=torch.float32)
    test_y = torch.tensor(test_data["y"], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    input_shape = (train_x.shape[1], 1)
    output_shape = train_y.shape[1]
    model = TransformerModel(input_shape, output_shape).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 10
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=batch_size)

    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
            print(f"Epoch {epoch+1} - Validation Loss: {val_loss/len(val_loader):.4f}")

        # Evaluate the model on the test set
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size)
        test_loss = 0
        predictions = []
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            test_loss += criterion(pred, batch_y).item()
            predictions.extend(pred.cpu().numpy())

        predictions = np.array(predictions)
        np.save("predictions.npy", predictions)
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")



if __name__ == "__main__":
    main()

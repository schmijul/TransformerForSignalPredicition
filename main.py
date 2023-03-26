import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.transformermodel import TransformerModel
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




def create_dataloader(data, batch_size=64):
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.view(-1, 1))
            running_loss += loss.item()
    return running_loss / len(dataloader)







if __name__ == "__main__":
    
    
    train_data, val_data, test_data = load_dataset()    
    batch_size = 2 * 64
    train_dataloader = create_dataloader(train_data, batch_size)
    val_dataloader = create_dataloader(val_data, batch_size)
    test_dataloader = create_dataloader(test_data, batch_size)
    input_dim = 25  # Anzahl der Eingabemerkmale ändern
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    output_dim = 1

    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    
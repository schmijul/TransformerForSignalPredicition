import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        """
        :param x: Input data (numpy.ndarray)
        :param y: Target data (numpy.ndarray)
        """
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        """
        :return: Length of the dataset (int)
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        :param idx: Index of the data point (int)
        :return: Tuple of input and target data (tuple)
        """
        return self.x[idx], self.y[idx]

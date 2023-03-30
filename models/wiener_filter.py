import numpy as np
from scipy.signal import wiener


class WienerFilter:
    def __init__(self, mysize=None, noise=None):
        self.mysize = mysize
        self.noise = noise

    def fit(self, signal):
        self.filter = wiener(signal, self.mysize, self.noise)

    def predict(self, signal):
        return wiener(signal, self.mysize, self.noise)

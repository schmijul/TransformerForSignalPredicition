import numpy as np
from scipy.signal import lfilter

class WienerFilter:
    def __init__(self, n):
        self.n = n
        self.autocorrelation = None
        self.R_matrix = None
        self.r_vector = None
        self.W = None

    def fit(self, signal):
        n = self.n
        self.autocorrelation = np.correlate(signal, signal, mode='full') / n
        self.autocorrelation = self.autocorrelation[n-1:]
        self.R_matrix = np.zeros((n, n))

        for i in range(n):
            self.R_matrix[i,:] = self.autocorrelation[n-i-1:n-i-1+n]

        self.r_vector = self.autocorrelation[:n]

        self.W = np.linalg.inv(self.R_matrix).dot(self.r_vector)

    def predict(self, signal):
        n = self.n
        predicted_value = lfilter(self.W, 1, signal[-n:])[0]
        return predicted_value

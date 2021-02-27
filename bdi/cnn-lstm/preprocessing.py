import numpy as np


class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def Split(x, y):
    a = int(0.8 * x.shape[0])
    b = int(0.9 * x.shape[0])
    return x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:]

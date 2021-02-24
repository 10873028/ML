import numpy as np


class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    
def Create_Matrix(data, timesteps):
    x, y = [], []
    for i in range(timesteps, data.shape[0]):
        x.append(data[i - timesteps:i])
        y.append(data[i])
    return np.array(x), np.array(y)


def Split(x, y, seed=0):
    a = int(0.8 * x.shape[0])
    b = int(0.9 * x.shape[0])
    index = np.arange(a)
    np.random.seed(seed)
    np.random.shuffle(index)
    return x[:a][index], y[:a][index], x[a:b], y[a:b], x[b:], y[b:]


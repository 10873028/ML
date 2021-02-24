import numpy as np


class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def Create_Matrix(mode, eemd, timesteps):
    x_train, y_train = [], []
    for i in range(timesteps, eemd[23][mode].shape[0]):
        x_train.append(eemd[23][mode, i - timesteps:i])
        y_train.append(eemd[23][mode, i])
    x_test, y_test = [], []
    for i in range(23, 1523):
        x_test.append(eemd[i][mode, -timesteps:])
        y_test.append(eemd[i + 1][mode, -1])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def Split(x_train, y_train, x_test, y_test, seed=42):
    a = np.arange(x_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(a)
    b = int(x_test.shape[0] * 0.5)
    return x_train[a], y_train[a], x_test[:b], y_test[:b], x_test[b:], y_test[b:]

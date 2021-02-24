import tensorflow as tf
import numpy as np

def forecast(data, model):
    a = data.shape[0]
    b = []
    for _ in range(data.shape[0]):
        predict = list(model.predict(data[-a:, 0].reshape(1, -1, 1)).reshape(-1))
        b.append(predict)
        data = np.array(list(data.reshape(-1)) + predict).reshape(-1, 1)
    return np.array(b)


class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class MinMaxScaler:
    def __init__(self, data):
        self.max = np.max(data)
        self.min = np.min(data)
    
    def transform(self, data):
        return (data - self.min) / self.max
    
    def inverse_transform(self, data):
        return data * self.max + self.min

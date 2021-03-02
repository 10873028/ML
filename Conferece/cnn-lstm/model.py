import torch
from torch import nn
import torch.nn.functional as func


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=32, num_layers=2, bidirectional=False, batch_first=True
        )
        self.sequential = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x, None)
        return self.sequential(out[:, -1, :])


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.cnn = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=2)
        self.maxpool = nn.MaxPool1d(2)
        self.ft = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(
            input_size=288, hidden_size=32, num_layers=2, bidirectional=False, batch_first=True
        )
        self.sequential = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, a):
        N, timesteps, C, L = a.shape
        a = a.view(N * timesteps, C, L)
        a = self.cnn(a)
        a = func.leaky_relu(a, 0.2)
        a = self.maxpool(a)
        a = a.view(N, timesteps, a.shape[1], a.shape[2])
        a = self.ft(a)
        a, _ = self.lstm(a, None)
        a = a[:, -1, :]
        return self.sequential(a)

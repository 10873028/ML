from torch import nn
import torch.nn.functional as func

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.maxpool = nn.MaxPool1d(2)
        self.ft = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=32, num_layers=1, bidirectional=False, batch_first=True
        )
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a):
        N, timesteps, C, L = a.shape
        a = a.view(N * timesteps, C, L)
        a = self.cnn1(a)
        a = func.leaky_relu(a, 0.2)
        a = self.cnn2(a)
        a = func.leaky_relu(a, 0.2)
        a = self.maxpool(a)
        a = a.view(N, timesteps, a.shape[1], a.shape[2])
        a = self.ft(a)
        a, _ = self.lstm(a, None)
        a = a[:, -1, :]
        a = self.fc(a)
        return self.sigmoid(a)
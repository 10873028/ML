from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=64, num_layers=2, bidirectional=False, batch_first=True
        )
        self.sequential = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x, _ = self.lstm(x, None)
        x = x[:, -1, :]
        return self.sequential(x)
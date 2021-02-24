from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=16, num_layers=1, bidirectional=True, batch_first=True
        ) 
        self.sequential = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        out, _ = self.lstm(x, None)
        return self.sequential(out[:, -1, :])
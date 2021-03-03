from torch import nn
import torch.nn.functional as func

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=32, num_layers=2, bidirectional=False, batch_first=True
        )
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x, None)
        x = self.fc1(x[:, -1, :])
        x = func.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
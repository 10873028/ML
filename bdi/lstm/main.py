import torch
from torch import nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from model import Net
from preprocessing import Create_Matrix, StandardScaler, Split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

timesteps = 90
batch_size = 512

data = pd.read_csv('bdi.csv', parse_dates=['date'], index_col=['date']).fillna(method='ffill').values

x, y = Create_Matrix(data, timesteps)
x_train, y_train, x_val, y_val, x_test, y_test = Split(x, y)

scaler = StandardScaler(y_train)
y_train = scaler.transform(y_train)
x_train = scaler.transform(x_train)
y_val = scaler.transform(y_val)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
x_val = torch.FloatTensor(x_val).to(device)
y_val = torch.FloatTensor(y_val).to(device)
x_test = torch.FloatTensor(x_test).to(device)

dataset = Data.TensorDataset(x_train, y_train)
dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = Net().to(device)
adam = torch.optim.Adam(model.parameters(), lr=0.01)
sgd = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 200
history = dict()
history['loss'] = []
history['val_loss'] = []
best_model = np.inf
for epoch in range(epochs):
    for x_batch, y_batch in dataloader:
        model.train()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        adam.zero_grad()
        loss.backward()
        adam.step()
    with torch.no_grad():
        model.eval()
        y_pred = model(x_val)
        val_loss = criterion(y_pred, y_val)
        loss = loss.item()
        val_loss = val_loss.item()
    if val_loss <= best_model:
        best_model = val_loss
        torch.save(model, f'checkpoint.pt')
        print('weight_saved')
    history['loss'].append(loss)
    history['val_loss'].append(val_loss)
    print(f'Epoch:{1+epoch:03d}-loss:{loss:.5f}-val_loss:{val_loss:.5f}')

model.eval()
with torch.no_grad():
    pred = model(x_test)
    pred = pred.cpu().numpy()
    pred = scaler.inverse_transform(pred)

plt.figure(figsize=(16, 8))
plt.plot(y_test, label='Actual')
plt.plot(pred, label='Predict')
plt.title(metrics.mean_absolute_error(pred, y_test))
plt.legend()
plt.savefig('fig/predict.jpg')
plt.show()

plt.figure(figsize=(16, 8))
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.title('History')
plt.legend()
plt.savefig('fig/history.jpg')
plt.show()

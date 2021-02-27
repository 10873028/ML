import torch
from torch import nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from preprocessing import StandardScaler, Split
from model import Classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

timesteps = 90
batch_size = 512
epochs = 300

df = pd.read_csv('bdi.csv', parse_dates=['date'], index_col=['date'])
df.fillna(method='ffill', inplace=True)

sample = df.values
label = [0]
for i in range(1, sample.shape[0]):
    if sample[i] > sample[i-1]:
        label.append(1)
    elif sample[i] < sample[i-1]:
        label.append(2)
    else:
        label.append(0)
label = np.array(label)

x = []
y = []
for i in range(timesteps, label.shape[0]):
    x.append(sample[i-timesteps:i])
    y.append(label[i])

x = np.array(x)
y = np.array(y)

x_train, y_train, x_val, y_val, x_test, y_test = Split(x, y)

oe = preprocessing.OneHotEncoder()
train = oe.fit_transform(y_train.reshape(-1, 1)).toarray()
train = pd.DataFrame(columns=['0', '1', '2'], data=train)
train.to_csv('train.csv', index=False, header=True)

scaler = StandardScaler(x_train.reshape(-1))
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(device)
x_val = torch.FloatTensor(x_val).to(device)
x_test = torch.FloatTensor(x_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_val = torch.LongTensor(y_val).to(device)

dataset = Data.TensorDataset(x_train, y_train)
dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = Classifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

best_model = np.inf
history = {}
history['loss'] = []
history['val_loss'] = []
for epoch in range(epochs):
    if epoch == 150:
        model = torch.load('checkpoint.pt')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for x_batch, y_batch in dataloader:
        model.train()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(x_val)
    val_loss = criterion(y_pred, y_val)
    history['loss'].append(loss.item())
    history['val_loss'].append(val_loss.item())
    print(f'epoch:{epoch + 1:03d}-loss:{loss.item():.5f}-val_loss:{val_loss.item():.5f}')
    if val_loss.item() <= best_model:
        best_model = val_loss.item()
        torch.save(model, 'checkpoint.pt')
        print('weight saved')

model = torch.load('checkpoint.pt')

with torch.no_grad():
    model.eval()
    predict = model(x_test).cpu().numpy()

predict = np.argmax(predict, axis=1)
print(metrics.accuracy_score(predict, y_test))

plt.figure(figsize=(16, 8))
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.title('History')
plt.legend()
plt.show()

with torch.no_grad():
    model.eval()
    test = model(x_test).cpu().numpy()
test = np.argmax(test, axis=1).reshape(-1, 1)
test = oe.transform(test).toarray()
test = pd.DataFrame(columns=['0', '1', '2'], data=test)
test.to_csv('test.csv', index=False, header=True)

with torch.no_grad():
    model.eval()
    val = model(x_val).cpu().numpy()
val = np.argmax(val, axis=1).reshape(-1, 1)
val = oe.transform(val).toarray()
val = pd.DataFrame(columns=['0', '1', '2'], data=val)
val.to_csv('val.csv', index=False, header=True)

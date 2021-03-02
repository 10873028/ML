import torch
from torch import nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import StandardScaler, Create_Matrix, Split
from model import Net
# 使用cuda或cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# 超參數
timesteps = 90
baggings = 20
batch_size = 512
epochs0 = 100
epochs1 = 30
# 匯入資料
eemd = [np.load(f'EEMD/EEMD{i}.npy').reshape(11, -1, 1) for i in range(1524)]
# 迴圈所有模態
for mode in range(int(input('start')), int(input('end'))):
    # 資料預處理(定義Sample與Label, 切分訓練、驗證、測試)
    x_train, y_train, x_test, y_test = Create_Matrix(mode, eemd, timesteps)
    x_train, y_train, x_val, y_val, x_test, y_test = Split(x_train, y_train, x_test, y_test)
    # 資料預處理(資料標準化)
    scaler = StandardScaler(y_train)
    y_train = scaler.transform(y_train)
    x_train = scaler.transform(x_train)
    y_val = scaler.transform(y_val)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    # 將array轉換為tensor
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    # 設定dataloader以batch進行訓練
    dataset = Data.TensorDataset(x_train, y_train)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # 訓練
    predict = 0
    plt.figure(figsize=(16, 8))
    for bagging in range(baggings):
        # 建立模型
        model = Net().to(device)
        print(model)
        # 設定loss, optimizer參數
        criterion = nn.MSELoss()
        adam = torch.optim.Adam(model.parameters(), lr=0.01)
        # 設定變數
        best_model = np.inf
        history = dict()
        history['loss'] = []
        history['val_loss'] = []
        for epoch in range(epochs0):
            model.train()
            for x_batch, y_batch in dataloader:
                # 輸出
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                # 清除梯度
                adam.zero_grad()
                # 反向傳播
                loss.backward()
                # 調整權重
                adam.step()
                # 驗證
            model.eval()
            with torch.no_grad():
                y_pred = model(x_val)
                val_loss = criterion(y_pred, y_val)
            loss = loss.item()
            val_loss = val_loss.item()
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            print(
                f'mode:{mode}-epoch:{epoch+1:03d}-bagging:{bagging:02d}-loss:{loss:.7f}-val_loss:{val_loss:.7f}')
            # 判斷是否為當前最佳模型
            if val_loss <= best_model:
                best_model = val_loss
                torch.save(model, f'mode{mode:02d}/{bagging}.pt')
                print('weights saved')
        # 讀取最佳模型
        model = torch.load(f'mode{mode:02d}/{bagging}.pt')
        # 設定optimizer參數
        sgd = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs1):
            model.train()
            for x_batch, y_batch in dataloader:
                # 輸出
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                # 清除梯度
                sgd.zero_grad()
                # 反向傳播
                loss.backward()
                # 調整權重
                sgd.step()
            model.eval()
            with torch.no_grad():
                y_pred = model(x_val)
                val_loss = criterion(y_pred, y_val)
            loss = loss.item()
            val_loss = val_loss.item()
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            print(
                f'mode:{mode}-epoch:{epoch+epochs0+1:03d}-bagging:{bagging:02d}-loss:{loss:.7f}-val_loss:{val_loss:.7f}')
            if val_loss <= best_model:
                best_model = val_loss
                torch.save(model, f'mode{mode:02d}/{bagging}.pt')
                print('weight saved')
        # 讀取最佳模型
        model = torch.load(f'mode{mode:02d}/{bagging}.pt')
        model.eval()
        with torch.no_grad():
            predict += model(x_test).cpu().numpy()

        plt.plot(history['loss'], label=f'loss-{bagging:02d}')
        plt.plot(history['val_loss'], label=f'val_loss-{bagging:02d}')
    plt.title(f'Mode{mode:02d}')
    plt.legend()
    plt.savefig(f'fig/Mode{mode:02d}-history.jpeg')
    plt.show()
    # 平均
    predict = predict / baggings
    predict = scaler.inverse_transform(predict)
    # 繪製圖表
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, label='Actual')
    plt.plot(predict, label='Predict')
    plt.title(f'Mode{mode:02d}-{metrics.mean_absolute_error(y_test, predict):.2f}')
    plt.legend()
    plt.savefig(f'fig/Mode{mode}-predict.jpeg')
    plt.show()
    # 儲存表單
    df = pd.read_csv('Ensemble.csv')
    df[f'Mode{mode:02d}'] = predict
    df.to_csv('Ensemble.csv', index=False, header=True)

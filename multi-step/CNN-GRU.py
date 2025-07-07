
import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader(batch_size, workers=0):
    train_set = load('train_xdata')
    train_label = load('train_ylabel')
    test_set = load('test_xdata')
    test_label = load('test_ylabel')

    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_set, train_label),
                                   batch_size=batch_size, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_set, test_label),
                                  batch_size=batch_size, num_workers=workers, drop_last=False)
    return train_loader, test_loader

batch_size = 32
train_loader, test_loader = dataloader(batch_size)

import torch
import torch.nn as nn

class CNNGRUModel(nn.Module):
    def __init__(self, batch_size, input_dim, conv_archs, hidden_layer_sizes, output_dim, output_size):

        super().__init__()
        self.batch_size = batch_size
        self.conv_arch = conv_archs
        self.input_channels = input_dim
        self.features = self.make_layers()

        self.num_layers = len(hidden_layer_sizes)
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(conv_archs[-1][-1], hidden_layer_sizes[0], batch_first=True))
        for i in range(1, self.num_layers):
            self.gru_layers.append(nn.GRU(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))
        self.linear = nn.Linear(hidden_layer_sizes[-1], output_dim * output_size)
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        # [batch, dim, seq_length]
        input_seq = input_seq.permute(0, 2, 1)
        cnn_features = self.features(input_seq)  # [batch, channels, seq_length]

        gru_input = cnn_features.permute(0, 2, 1)
        for gru in self.gru_layers:
            gru_input, _ = gru(gru_input)
        predict = self.linear(gru_input[:, -1, :])
        return predict

batch_size = 32
input_dim = 23

conv_archs = ((1, 32), (1, 64))
hidden_layer_sizes = [64, 128]
output_dim = 1
output_size = 6

model = CNNGRUModel(batch_size, input_dim, conv_archs, hidden_layer_sizes, output_dim, output_size)

loss_function = nn.MSELoss(reduction='sum')  # loss
learn_rate = 0.0003
optimizer = torch.optim.Adam(model.parameters(), learn_rate)

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

count_parameters(model)
print(model)

import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='Microsoft YaHei')


def model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, test_loader):
    model = model.to(device)

    train_size = len(train_loader) * batch_size
    test_size = len(test_loader) * batch_size

    minimum_mse = 1000.
    best_model = model

    train_mse = []
    test_mse = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        train_mse_loss = 0.
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()

            y_pred = model(seq)  # torch.Size([16, 10])

            loss = loss_function(y_pred, labels)
            train_mse_loss += loss.item()
            loss.backward()
            optimizer.step()
        #     break
        # break
        train_av_mseloss = train_mse_loss / train_size
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch + 1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')

        with torch.no_grad():
            test_mse_loss = 0.
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)

                test_loss = loss_function(pre, label)
                test_mse_loss += test_loss.item()

            test_av_mseloss = test_mse_loss / test_size
            test_mse.append(test_av_mseloss)
            print(f'Epoch: {epoch + 1:2} test_MSE_Loss:{test_av_mseloss:10.8f}')
            if test_av_mseloss < minimum_mse:
                minimum_mse = test_av_mseloss
                best_model = model

    torch.save(best_model, 'best_model_cnn_gru.pt')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    plt.plot(range(epochs), train_mse, color='b', label='train_MSE-loss')
    plt.plot(range(epochs), test_mse, color='y', label='test_MSE-loss')
    plt.legend()
    plt.show()
    print(f'min_MSE: {minimum_mse}')

epochs = 1500
model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('best_model_cnn_gru.pt',weights_only=False)
model = model.to(device)

original_data = []
pre_data = []
with torch.no_grad():
        for data, label in test_loader:
            origin_lable = label.tolist()
            original_data += origin_lable

            model.eval()
            data, label = data.to(device), label.to(device)
            test_pred = model(data)
            test_pred = test_pred.tolist()
            pre_data += test_pred

original_data = np.array(original_data)
pre_data = np.array(pre_data)
print('shape：')
print(original_data.shape, pre_data.shape)

scaler  = load('scaler ')
original_data = scaler.inverse_transform(original_data)
pre_data = scaler.inverse_transform(pre_data)
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
batch_size = 32

model = torch.load('best_model_cnn_gru.pt')
model = model.to(device)

scaler = load('scaler ')

score = r2_score(original_data, pre_data)
print('R^2:',score)
test_mse = mean_squared_error(original_data, pre_data)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(original_data, pre_data)
print('MSE: ',test_mse)
print('RMSE: ',test_rmse)
print('MAE: ',test_mae)
def nse(original_data, pre_data):

    numerator = np.sum((original_data - pre_data) ** 2)
    denominator = np.sum((original_data - np.mean(original_data)) ** 2)
    nse_value = 1 - (numerator / denominator)
    return nse_value

nse_value = nse(original_data, pre_data)
print(f'NSE: {nse_value}')
def wbe(original_data, pre_data, weights):

    weighted_bias = np.sum(weights * (pre_data - original_data))
    weighted_sum = np.sum(weights)
    wbe_value = weighted_bias / weighted_sum
    return wbe_value

weights = np.ones_like(original_data)
wbe_value = wbe(original_data, pre_data, weights)
print(f'WBE: {wbe_value}')

def model_prediction(model, data, batch_size, scaler):

    temp_data = data.repeat(batch_size, 1, 1)  # torch.Size([64, 24, 1])

    temp_data = temp_data.to(device)
    my_pre = model(temp_data)

    my_pre = np.array(my_pre.tolist())

    my_pre = scaler.inverse_transform(my_pre)

    pre = my_pre[0]
    return pre

testset = load('test_xdata')  # torch.Size([3470, 24, 1])
my_data = testset[-1]  # torch.Size([24, 1])

pre = model_prediction(model, my_data, batch_size, scaler)
print(pre)
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

forecast_step = 6

labels = []
for i in range(forecast_step):
    label = f"T + {i+1} step"
    labels.append(label)
step = 5

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(original_data[:, step], label='origin signal',color='orange')
plt.plot(pre_data[:, step], label=f'CNN-LSTM model T + {step + 1} step predict',color='green')
plt.legend()
plt.show()

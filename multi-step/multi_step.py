
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

print(len(train_loader))
print(len(test_loader))

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):

        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):

        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell

class Seq2SeqLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pre_len):

        super(Seq2SeqLSTMModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)
        self.pre_len = pre_len

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_vocab_size = trg.size(2)
        outputs = torch.zeros(batch_size, self.pre_len, trg_vocab_size).to(src.device)

        hidden, cell = self.encoder(src)

        input = src[:, -1, -1:].unsqueeze(1)

        for t in range(self.pre_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(-1)

            input = output if np.random.random() > teacher_forcing_ratio else trg[:, t, :].unsqueeze(1)


        return outputs.squeeze(2)

input_dim = 23
hidden_dim = 128
num_layers = 2
output_dim = 1
pre_len = 6

model = Seq2SeqLSTMModel(input_dim, hidden_dim, num_layers, output_dim, pre_len)

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


def model_train(epochs, model, optimizer, loss_function, train_loader, test_loader, device):
    model = model.to(device)

    minimum_mse = 1000.
    best_model = model

    train_mse = []
    test_mse = []

    start_time = time.time()
    for epoch in range(epochs):

        model.train()

        train_mse_loss = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()

            decoder_input = labels.unsqueeze(2)
            y_pred = model(seq, decoder_input)

            loss = loss_function(y_pred, labels)
            train_mse_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_av_mseloss = np.average(train_mse_loss)
        train_mse.append(train_av_mseloss)

        print(f'Epoch: {epoch + 1:2} train_MSE-Loss: {train_av_mseloss:10.8f}')

        with torch.no_grad():

            model.eval()
            test_mse_loss = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                decoder_input = label.unsqueeze(2)
                pre = model(data, decoder_input)

                test_loss = loss_function(pre, label)
                test_mse_loss.append(test_loss.item())

            test_av_mseloss = np.average(test_mse_loss)
            test_mse.append(test_av_mseloss)
            print(f'Epoch: {epoch + 1:2} test_MSE_Loss:{test_av_mseloss:10.8f}')

            if test_av_mseloss < minimum_mse:
                minimum_mse = test_av_mseloss
                best_model = model

    torch.save(best_model, 'best_model_Seq2SeqLSTMModel.pt')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    plt.plot(range(epochs), train_mse, color='b', label='train_MSE-loss')
    plt.plot(range(epochs), test_mse, color='y', label='test_MSE-loss')
    plt.legend()
    plt.show()
    print(f'min_MSE: {minimum_mse}')

epochs = 1500
model_train(epochs, model, optimizer, loss_function, train_loader, test_loader, device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('best_model_Seq2SeqLSTMModel.pt')
model = model.to(device)

original_data = []
pre_data = []
with torch.no_grad():
        for data, label in test_loader:
            origin_lable = label.tolist()
            original_data += origin_lable

            model.eval()
            data, label = data.to(device), label.to(device)

            decoder_input = label.unsqueeze(2)
            test_pred = model(data, decoder_input)
            test_pred = test_pred.tolist()
            pre_data += test_pred

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error

original_data = np.array(original_data)
pre_data = np.array(pre_data)
print('data shape：')
print(original_data.shape, pre_data.shape)

scaler  = load('scaler')
original_data = scaler.inverse_transform(original_data)
pre_data = scaler.inverse_transform(pre_data)

score = r2_score(original_data, pre_data)
print('R^2:',score)

test_mse = mean_squared_error(original_data, pre_data)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(original_data, pre_data)
mape = mean_absolute_percentage_error(original_data, pre_data)
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
model = torch.load('best_model_Seq2SeqLSTMModel.pt',weights_only=False)
model = model.to(device)

forecast_step = 6


def model_prediction(model, data, batch_size, scaler, forecast_step):

    temp_data = data.repeat(batch_size, 1, 1)  # torch.Size([64, 24, 1])

    temp_data = temp_data.to(device)
    decoder_input = temp_data[:, -forecast_step:, -1:]
    my_pre = model(temp_data, decoder_input)

    my_pre = np.array(my_pre.tolist())

    my_pre = scaler.inverse_transform(my_pre)
    pre = my_pre[0]
    return pre

testset = load('test_xdata')  # torch.Size([3470, 12, 7])
my_data = testset[-1]  # torch.Size([12, 7])

pre = model_prediction(model, my_data, batch_size, scaler, forecast_step)
print(pre)

import numpy as np
import matplotlib.pyplot as plt

labels = []
for i in range(forecast_step):
    label = f"T + {i+1} step predict"
    labels.append(label)

step = 0

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(original_data[:, step], label='origin signal',color='orange')
plt.plot(pre_data[:, step], label=f'Seq2Seq model T + {step + 1} step predict',color='green')
plt.legend()
plt.show()



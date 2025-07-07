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
                                  batch_size=batch_size, num_workers=workers, drop_last=True)
    return train_loader, test_loader

batch_size = 64
train_loader, test_loader = dataloader(batch_size)

print(len(train_loader))
print(len(test_loader))

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = self.v(energy).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        weighted = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        weighted = weighted.squeeze(1)
        return weighted, attention_weights
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.lstm = nn.LSTM(output_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, x, hidden, cell, encoder_outputs):

        weighted, attention_weights = self.attention(hidden[-1], encoder_outputs)

        lstm_input = torch.cat((x, weighted.unsqueeze(1)), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        predictions = self.fc(outputs)
        return predictions, hidden, cell, attention_weights
class Seq2SeqLSTMModelWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pre_len):
        super(Seq2SeqLSTMModelWithAttention, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = DecoderWithAttention(hidden_dim, output_dim, num_layers)
        self.pre_len = pre_len

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_vocab_size = trg.size(2)
        outputs = torch.zeros(batch_size, self.pre_len, trg_vocab_size).to(src.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)

        input = src[:, -1, -1:].unsqueeze(1)

        for t in range(self.pre_len):
            output, hidden, cell, attention_weights = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output.squeeze(-1)
            input = output if np.random.random() > teacher_forcing_ratio else trg[:, t, :].unsqueeze(1)

        return outputs.squeeze(2)
input_dim = 23
hidden_dim = 128
num_layers = 2
output_dim = 1
pre_len = 6

model = Seq2SeqLSTMModelWithAttention(input_dim, hidden_dim, num_layers, output_dim, pre_len)

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

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def k_fold_cross_validation(k, epochs, model, optimizer, loss_function, train_set, batch_size, device):
    # KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=50)

    fold_train_mse = []
    fold_test_mse = []
    best_models = []

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
        print(f"Fold {fold + 1}/{k}")

        train_subsampler = Subset(train_set, train_idx)
        val_subsampler = Subset(train_set, val_idx)

        ktrain_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

        model = model.to(device)
        optimizer.zero_grad()

        minimum_mse = float('inf')
        best_model = None

        train_mse = []
        test_mse = []

        for epoch in range(epochs):
            model.train()
            train_mse_loss = []
            for seq, labels in ktrain_loader:
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

            print(f'Epoch {epoch + 1}/{epochs} Train Mseloss: {train_av_mseloss:.8f}')

            model.eval()
            val_mse_loss = []
            with torch.no_grad():
                for data, label in val_loader:
                    data, label = data.to(device), label.to(device)
                    decoder_input = label.unsqueeze(2)
                    pre = model(data, decoder_input)

                    val_loss = loss_function(pre, label)
                    val_mse_loss.append(val_loss.item())

            val_av_mseloss = np.average(val_mse_loss)
            test_mse.append(val_av_mseloss)

            print(f'Epoch {epoch + 1}/{epochs} Val Mseloss: {val_av_mseloss:.8f}')

            if val_av_mseloss < minimum_mse:
                minimum_mse = val_av_mseloss
                best_model = model

        best_models.append(best_model)
        fold_train_mse.append(np.average(train_mse))
        fold_test_mse.append(np.average(test_mse))

        print(f"Best MSE for Fold {fold + 1}: {minimum_mse:.8f}")

    avg_train_mse = np.average(fold_train_mse)
    avg_test_mse = np.average(fold_test_mse)
    print(f"\nAverage Train MSE: {avg_train_mse:.8f}")
    print(f"Average Test MSE: {avg_test_mse:.8f}")

    torch.save(best_models, 'best_models_lstmseqattention.pt')

    print(f"\nDuration: {time.time() - start_time:.0f} seconds")

    plt.plot(range(epochs), train_mse, color='b', label='train_MSE-loss')
    plt.plot(range(epochs), test_mse, color='y', label='test_MSE-loss')
    plt.legend()
    plt.show()
    print(f'min_MSE: {minimum_mse}')

k = 5
batch_size = 64
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = dataloader(batch_size)
dataset = train_loader.dataset

k_fold_cross_validation(k, epochs, model, optimizer, loss_function, dataset, batch_size, device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('best_models_lstmseqattention.pt')
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
print('shape：')
print(original_data.shape, pre_data.shape)

scaler  = load('scaler')
original_data = scaler.inverse_transform(original_data)
pre_data = scaler.inverse_transform(pre_data)

score = r2_score(original_data, pre_data)
print('*'*50)
print('R^2:',score)

print('*'*50)

test_mse = mean_squared_error(original_data, pre_data)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(original_data, pre_data)
mape = mean_absolute_percentage_error(original_data, pre_data)
print(f'MAPE: {mape * 100:.2f}%')
print('MSE: ',test_mse)
print('RMSE: ',test_rmse)
print('MAE: ',test_mae)

model = torch.load('best_model_seq2seq.pt',weights_only=False)
model = model.to(device)

scaler = load('scaler')
batch_size = 64
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
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

forecast_step = 6

labels = []
for i in range(forecast_step):
    label = f"T + {i+1} step"
    labels.append(label)

step = 0

plt.figure(figsize=(12, 6), dpi=100)
plt.plot(original_data[:, step], label='origin signal',color='orange')
plt.plot(pre_data[:, step], label=f'Seq2Seq model T + {step + 1} step predict',color='green')
plt.legend()
plt.show()

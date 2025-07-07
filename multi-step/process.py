import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

file_path = "Ganzhou(process)_month_VMD.xlsx"

original_data = pd.read_excel(file_path)
print(original_data.shape)
original_data.head()
OTddata = original_data['PRCP']

plt.figure(figsize=(15,5), dpi=100)
plt.grid(True)
plt.plot(OTddata, color='green')
plt.show()
import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler

OTddata = np.array(OTddata)
OTddata = OTddata.reshape(-1, 1)

var_data =  original_data[['DEWP','SLP','WDSP','MAX','MIN','PRCP']]
# YiLi
#var_data =  original_data[['E','average','max','min','IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9','IMF10','IMF11','IMF12','IMF13','IMF14','IMF15','IMF16','IMF17','IMF18']]

#LiuTing
#var_data =  original_data[['TEMP', 'DEWP', 'WDSP', 'MXSPD', 'MAX','IMF1','IMF2','IMF3','IMF4','IMF5','IMF6','IMF7','IMF8','IMF9','IMF10','IMF11','IMF12','IMF13','IMF14','IMF15','IMF16','IMF17','IMF18']]
ylable_data =  OTddata

scaler = MinMaxScaler()
var_data = scaler.fit_transform(var_data)
ylable_data = scaler.fit_transform(ylable_data)

dump(scaler, 'scaler')
def create_multistep_dataset(x_var, ylable_data, window_size, forecast_step):

    sample_features = []
    labels = []
    for i in range(len(ylable_data) - window_size - forecast_step + 1):
        sample_features.append(x_var[i:i + window_size, :])
        labels.append(ylable_data[i + window_size:i + window_size + forecast_step])

    sample_features = np.array(sample_features)
    labels = np.array(labels)
    labels = labels.reshape(-1, forecast_step)
    sample_features = torch.tensor(sample_features).float()
    labels = torch.tensor(labels).float()
    return sample_features, labels

def make_dataset(var_data, ylable_data, window_size, forecast_step, split_rate=[0.7, 0.3]):

    sample_len = var_data.shape[0]
    train_len = int(sample_len * split_rate[0])
    train_var = var_data[:train_len, :]
    test_var = var_data[train_len:, :]

    train_y = ylable_data[:train_len]
    test_y = ylable_data[train_len:]

    train_set, train_label = create_multistep_dataset(train_var, train_y, window_size, forecast_step)
    test_set, test_label = create_multistep_dataset(test_var, test_y, window_size, forecast_step)

    return train_set, train_label, test_set, test_label

window_size = 12
forecast_step = 6

split_rate = [0.8, 0.2]

train_xdata, train_ylabel, test_xdata, test_ylabel = make_dataset(var_data, ylable_data, window_size, forecast_step,
                                                                  split_rate)
dump(train_xdata, 'train_xdata')
dump(test_xdata, 'test_xdata')
dump(train_ylabel, 'train_ylabel')
dump(test_ylabel, 'test_ylabel')
print('data shape：')
print(train_xdata.size(), train_ylabel.size())
print(test_xdata.size(), test_ylabel.size())


from model import *
from train_utilities import *
from general_utilities import *
from baseline_models import *
from extreme_time(EVL) import *
from extreme_time_2(EVL) import *
from train_model_gev import *


from pandas import DataFrame
import pandas as pd
import numpy as np
import os, random
import math
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import rc, style
import seaborn as sns
import datetime as dt
from tqdm import tqdm as tq
from numpy import vstack, sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas.plotting import register_matplotlib_converters
import matplotlib.patches as mpatches
from google.colab import files
from statistics import mean
import scipy.stats as stats
from scipy.special import gamma
import numpy.ma as ma

from scipy.stats import genextreme
from scipy.stats import pearsonr

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from functools import partial
from pylab import rcParams

import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.stopper import TrialPlateauStopper

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.8)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 24, 10
register_matplotlib_converters()

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING=1
torch.use_deterministic_algorithms(False)

# Data Loading

data_path = '/content/drive/MyDrive/PhD-Research-Phase-1/Data/'
data_file = data_path+'deseasonedMonthly_smallZscore_new.mat'
data = scipy.io.loadmat(data_file)

xMonth = data["deszXMonth"]
yMonth = data["deszYMonth"]
xMonth.shape, yMonth.shape, xMonth[0], yMonth[0]

# Data Creation

total_timesteps= 24
train_time_steps = 12
test_time_steps = total_timesteps - train_time_steps
number_of_locations = yMonth.shape[0]

train_test_ratio = int(number_of_locations * 0.95)
train_model_ratio = int(train_test_ratio * 0.80)
train_test_ratio, train_model_ratio, train_test_ratio - train_model_ratio

train_data_raw = []
test_data_raw = []
model_train_data_raw = []
model_test_data_raw = []

for i in range(number_of_locations):
  temp_data = yMonth[i,:,0]
  temp_data_model = xMonth[i,:,:]
  for j in range(0, temp_data.shape[0], 24):
    if temp_data.shape[0] - j < 24: break
    elif i < train_test_ratio:
      train_data_raw.append(temp_data[j:j+total_timesteps])
      if i >= train_model_ratio: model_train_data_raw.append(temp_data_model[j+train_time_steps:j+total_timesteps][:])
    else:
      test_data_raw.append(temp_data[j:j+total_timesteps])
      model_test_data_raw.append(temp_data_model[j+train_time_steps:j+total_timesteps][:])

train_data_raw= np.array(train_data_raw).reshape(-1,total_timesteps)
test_data_raw = np.array(test_data_raw).reshape(-1,total_timesteps)
model_train_data_raw= np.array(model_train_data_raw).reshape(-1, 13, test_time_steps)
model_test_data_raw = np.array(model_test_data_raw).reshape(-1, 13, test_time_steps )
train_data_raw.shape, test_data_raw.shape, model_train_data_raw.shape, model_test_data_raw.shape

# Train, Validate, Test Splits
# print("Train and Test Data shape before normalizing/standardizing:", train_m_data.shape, test_m_data.shape)
scaler=StandardScaler()
train_m_data=scaler.fit_transform(model_train_data_raw.reshape(-1,1))

train_m_data = train_m_data.reshape(-1,13, test_time_steps)
print("Train Data shape after normalizing/standardizing:", train_m_data.shape)

test_m_data=scaler.transform(model_test_data_raw.reshape(-1,1))

test_m_data = test_m_data.reshape(-1,13, test_time_steps)
print("Train and Test Data shape before normalizing/standardizing:", train_m_data.shape, test_m_data.shape)
print("Mean of train_m and test_m data", np.mean(train_m_data), np.mean(test_m_data))
train_m_only_data = train_m_data
val_m_only_data = train_m_data[0:130]
test_m_only_data = test_m_data

h_only_shape = train_data_raw.shape[0] -  model_train_data_raw.shape[0]
combined_shape = train_data_raw.shape[0]-h_only_shape
h_only_shape, combined_shape

h_data_only = train_data_raw[0:h_only_shape]
h_combined_data = train_data_raw[h_only_shape:]
train_m_data = model_train_data_raw
test_m_data = model_test_data_raw
train_data_raw.shape, test_data_raw.shape, model_train_data_raw.shape, model_test_data_raw.shape

np.mean(train_m_data), np.mean(test_m_data)
m_mask_data = np.zeros((h_data_only.shape[0], 13, test_time_steps))

m_mean = np.ones(train_m_data.shape[2])
m_std = np.ones(train_m_data.shape[1])
for i in range(train_m_data.shape[2]):
    m_mean[i] = np.mean(train_m_data[:, :, i])
for i in range(train_m_data.shape[1]):
    m_std[i] = np.std(train_m_data[:, i, :])

for i in range(h_data_only.shape[0]):
    for j in range(train_m_data.shape[1]):
        for k in range(train_m_data.shape[2]):
            m_mask_data[i][j][k] = np.random.normal(m_mean[k], m_std[j], 1)

train_h_data = np.concatenate((h_combined_data, h_data_only), axis=0)
test_h_data = test_data_raw
train_m_data = np.concatenate((train_m_data, m_mask_data), axis=0)
test_m_data = test_m_data
train_mask_data = np.zeros((train_h_data.shape[0], train_m_data.shape[1]))
train_mask_data[0:combined_shape][:] = 1
test_mask_data = np.ones((test_m_data.shape[0], test_m_data.shape[1]))
train_h_data.shape, train_m_data.shape, test_h_data.shape, test_m_data.shape, train_mask_data.shape, test_mask_data.shape

print("History data:")
# scaler=MinMaxcaler(feature_range=(0,1))
scaler=StandardScaler()
train_h_data=scaler.fit_transform(train_h_data.reshape(-1,1))

train_h_data = train_h_data.reshape(-1,total_timesteps)
print("Train Data shape after normalizing/standardizing:", train_h_data.shape)

test_h_data=scaler.transform(test_h_data.reshape(-1,1))

test_h_data = test_h_data.reshape(-1,total_timesteps)
print("Train and Test Data shape before normalizing/standardizing:", train_h_data.shape, test_h_data.shape)

print("Before Validation Data: train vs test", train_h_data.shape, test_h_data.shape)

length = 128
# random.shuffle(train_h_data)
val_h_data= train_h_data[0:length]
print("After Validation Data (from train data): train vs validation vs test", train_h_data.shape, val_h_data.shape, test_h_data.shape)

print("Model data:")
# random.shuffle(train_m_data)
val_m_data= train_m_data[0:length]
val_mask_data = np.ones((val_h_data.shape[0], val_m_data.shape[1]))
print("After Validation Data (from train data): train vs validation vs test", train_m_data.shape, val_m_data.shape, test_m_data.shape)

#Data Preprocessing

batch_size = 128

X_train_h, X_val_h, X_test_h = ready_X_data(train_h_data, val_h_data, test_h_data, train_time_steps)
X_train_m, X_val_m, X_test_m = ready_X_m_data(train_m_data, val_m_data, test_m_data, test_time_steps)
y_train , y_val, y_test  = ready_y_data(train_h_data, val_h_data, test_h_data, train_time_steps)

X_train_h, X_train_m, X_train_mask, y_train = extend_last_batch(X_train_h, X_train_m, train_mask_data, y_train)
X_val_h, X_val_m, X_val_mask, y_val = extend_last_batch(X_val_h, X_val_m, val_mask_data, y_val)
X_test_h, X_test_m, X_test_mask, y_test = extend_last_batch(X_test_h, X_test_m, test_mask_data, y_test)

X_train_h.shape, X_train_m.shape, X_train_mask.shape, y_train.shape, X_test_h.shape, X_test_m.shape, X_test_mask.shape, y_test.shape


temp_train_h = h_of_combined_data[:train_m_only_data.shape[0]]
temp_val_h = h_of_combined_data[:val_m_only_data.shape[0]]
temp_test_h = h_of_combined_data[-test_m_only_data.shape[0]:]
temp_train_h.shape, temp_test_h.shape

X_train_m_only, X_val_m_only, X_test_m_only = ready_X_m_data(train_m_only_data, val_m_only_data, test_m_only_data, test_time_steps)
y_train_m_only , y_val_m_only, y_test_m_only  = ready_y_data(temp_train_h, temp_val_h, temp_test_h, train_time_steps)

X_train_m_only, y_train_m_only = extend_last_batch_m(X_train_m_only, y_train_m_only)
X_val_m_only,y_val_m_only = extend_last_batch_m( X_val_m_only, y_val_m_only)
X_test_m_only, y_test_m_only = extend_last_batch_m( X_test_m_only, y_test_m_only)

X_train_m_only.shape, y_train_m_only.shape, X_test_m_only.shape, y_test_m_only.shape

# Plotting Data and Learning/Checking global mu, sigma, xi using y

plot_histogram(y_train, plot_name = "y")



shape, loc, scale = genextreme.fit(y_train.cpu())
print(f"Scipy Estimated GEV Parameters: mu: {loc}, sigma: {scale}, xi: {- shape}")
calculate_nll(y_train.cpu(), torch.tensor(loc), torch.tensor(scale), torch.tensor(-shape), name= "Scipy estimated parameters")

X_dummy = torch.ones(y_train.shape[0])
X_dummy, y_truth = create_gev_data(mu =loc, sigma=scale, xi=-shape, size = y_train.shape[0])
# y_generated =scaler.inverse_transform(y_truth.reshape(-1,1))
plot_histogram(y_truth, plot_name="Generated y")


#Training and Results

torch.use_deterministic_algorithms(False)

batch_size = 64
sequence_len_h = train_time_steps
sequence_len_m = test_time_steps
n_features_h = 1
n_features_m = 21
n_hidden_h = 10
n_hidden_m = 15
n_layers = 3

#persistent
y_persistent_hat = torch.zeros(X_test_h.shape[0],1).to(device)
for i, x in enumerate(y_test):
  y_persistent_hat[i] = torch.max(X_test_h[i][-test_time_steps:])
print("RMSE of y (standardized): ", ((y_test - y_persistent_hat) ** 2).mean().sqrt().item())
print(y_persistent_hat.shape, y_test.shape)
inverted_y, inverted_yhat = inverse_scaler(y_test.tolist(), y_persistent_hat.tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_persistent_hat,y_test))
# print("Correlation between actual and Predicted (mean): ", calculate_corr(np.array(inverted_y), np.array(inverted_yhat)))
plot_scatter(inverted_y, inverted_yhat, model_name="Persistence")

#mean_of_max
y_model_mean_of_max_hat = torch.zeros(X_test_h.shape[0],1).to(device)
for i, x in enumerate(y_test):
  max_of_models = torch.zeros(X_test_m.shape[2],1).to(device)
  for j in range(X_test_m.shape[2]):
      max_of_models[j] =  torch.max(X_test_m[i,:,j])
  y_model_mean_of_max_hat[i] = torch.mean(max_of_models)
print("RMSE of y (standardized): ", ((y_test - y_model_mean_of_max_hat) ** 2).mean().sqrt().item())
print(y_model_mean_of_max_hat.shape, y_test.shape)
inverted_y, inverted_yhat = inverse_scaler(y_test.tolist(), y_model_mean_of_max_hat.tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_model_mean_of_max_hat,y_test))
# print("Correlation between actual and Predicted (mean): ", calculate_corr(np.array(inverted_y), np.array(inverted_yhat)))
plot_scatter(inverted_y, inverted_yhat, model_name="Persistence")



#LSTM-h
count_constraint_violation = []
lr = 0.001
# n_layers = 3
num_epochs = 30
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.0, lambda_2=0.5, model_name = "LSTM_GEV", tuning=False, validation=True, hetero = "h")
plot_losses(train_history, validation_history, test_history)
#@title
print("RMSE of y (standardized): ", ((y_all - yhat_all) ** 2).mean().sqrt().item())
inverted_y, inverted_yhat = inverse_scaler(y_all.tolist(), yhat_all.tolist())
print("RMSE of y : ", math.sqrt(mean_squared_error(inverted_y,inverted_yhat)))
print("Correlation between actual and Predicted (mean): ", calculate_corr(y_all, yhat_all))
plot_scatter(inverted_y, inverted_yhat, model_name="Model LSTM: y estimations")


#PM1-h+m
n_hidden_h = 8
n_layers = 5
count_constraint_violation = []
lr = 0.001
# n_hidden_h = 10
# n_layers = 3
num_epochs = 30
boundary_tolerance = 0.1
train_history = [0] * num_epochs
validation_history = [0] * num_epochs
test_history = [0] * num_epochs

model_fcn_all, mu_hat_all, sigma_hat_all, xi_hat_all, y_all, yhat_all, y_q1_all, y_q2_all = train_model_gev(lambda_ = 0.1, lambda_2=0.9, model_name = "PM1", tuning=False, validation=True, hetero="h+m")
plot_losses(train_history, validation_history, test_history)
calculate_nll(y_test, mu_hat_all, sigma_hat_all, xi_hat_all, name = "Model M3 Estimation (y)")
all_result(y_all.cpu(), yhat_all.cpu(), y_q1_all.cpu(), y_q2_all.cpu(), mu_hat_all.cpu(), sigma_hat_all.cpu(), xi_hat_all.cpu(), model_name="PM1")
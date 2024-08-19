from torch.utils.data import Dataset, DataLoader, random_split
from chinese_calendar import is_workday
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn as nn
import pandas as pd
import akshare as ak
import numpy as np
import datetime
import random
import torch
import json
import csv
import os
import time

import model as my_model

os.environ['NO_PROXY'] = 'finance.sina.com.cn'

device = torch.device("cpu")
batch_size = 65536
input_size = 121
hidden_size = 512
n_head = 2
output_size = 2
num_hidden_layers = 20
learning_rate = 2e-3
train_num_epochs = 10
max_load_data = 20000000
# max_load_data = 65536
weight_decay_rate = 0.0
model_type = "MLP"  # MLP, MLP_Residual, MLP_Tanh, RNN, GRU, LSTM, Transformer
model_name = 'MLP_e_2_tiran_79.02_test_81.19_val_83.57_bs_65536_as_1.0_is_121_hs_512_nh_2_os_2_nhl_20_lr_0.002_wd_0.0_ds_20000000_model_weights.bin'
index_name_list = ["sh000001", "sz399808"]

def pre_process(get_roll_yield_bar_df):
    exponential_list = []
    label = -1
    for i in range(len(get_roll_yield_bar_df)):
        nowday = datetime.datetime.strptime(get_roll_yield_bar_df.iloc[i][0], '%Y-%m-%d %H:%M:%S')
        if nowday.month == today.month and nowday.day == today.day:
            if (i + 119) <= len(get_roll_yield_bar_df):
                nowday_noon = datetime.datetime.strptime(get_roll_yield_bar_df.iloc[i + 119][0], '%Y-%m-%d %H:%M:%S')
                if nowday.hour == 9 and nowday.minute == 31 and nowday_noon.hour == 11 and nowday_noon.minute == 30:
                    exponential_list.extend(get_roll_yield_bar_df.iloc[i - 1:i + 120, 4])
                    if (i + 238) <= len(get_roll_yield_bar_df):
                        nowday_pm = datetime.datetime.strptime(get_roll_yield_bar_df.iloc[i + 238][0], '%Y-%m-%d %H:%M:%S')
                        if nowday_pm.hour == 15 and nowday_pm.minute == 0:
                            if float(exponential_list[0]) > float(get_roll_yield_bar_df.iloc[i + 238][4]):
                                label = 1
                            else:
                                label = 0
            break
    if len(exponential_list) == 121:
        pre_process_exponential = []
        base = float(exponential_list[0])
        for j in range(0, 121):
            index = float(exponential_list[j])
            pre_process_exponential.append(int((index - base) * 1000000 / base))

        exponential_np = np.array(pre_process_exponential)
        exponential_np.reshape(1, 121)
        exponential_tensor = torch.tensor(exponential_np, dtype=torch.float32)
        is_valid = True
    else:
        print(f"len {len(exponential_list)} exponential_tensor {exponential_list}")
        exponential_tensor = torch.tensor([-1]*121, dtype=torch.float32)
        is_valid = False

    return is_valid, exponential_tensor, exponential_list, label

def isTradeDay(date):
    if is_workday(date):
        if date.isoweekday() < 6:
            return True
    return False

if __name__ == '__main__':
    msg = ""
    i = 0
    today = datetime.date.today()
    today = today - datetime.timedelta(days=2)
    if isTradeDay(today):
        msg = msg + str(today) + "\n"
        model = my_model.init_model(model_type, input_size, hidden_size, output_size, num_hidden_layers, n_head, device)
        model.load_state_dict(torch.load('./model/' + model_name))

        for index_name in index_name_list:
            get_roll_yield_bar_df = ak.stock_zh_a_minute(symbol=index_name, period="1")
            is_valid, exponential_tensor, exponential_list, label = pre_process(get_roll_yield_bar_df)
            if is_valid:
                outputs = model(exponential_tensor)
                result = torch.softmax(outputs[0], dim=0)
                _, predicted = torch.max(outputs[0], 0)
                is_coincident = False
                if (result[0] > result[1] and exponential_tensor[-1] > 0) or (result[0] < result[1] and exponential_tensor[-1] < 0):
                    is_coincident = True
                if label != -1:
                    tmp_msg = (f"{i} {result[0] * 100:0.2f}% {result[1] * 100:0.2f}% "
                           f"{exponential_tensor[-1] / 10000:0.2f}% {is_coincident} {label==predicted}\n")
                else:
                    tmp_msg = (f"{i} {result[0] * 100:0.2f}% {result[1] * 100:0.2f}% "
                           f"{exponential_tensor[-1] / 10000:0.2f}% {is_coincident}\n")
            else:
                tmp_msg = f"{i} NA% NA%\n"
            msg = msg + tmp_msg
            i = i + 1
        print(msg)
    else:
        print(f"today {str(today)} is not TradeDay")


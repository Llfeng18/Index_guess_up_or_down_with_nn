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
output_size = 2
num_hidden_layers = 20
learning_rate = 2e-3
num_epochs = 100
max_load_data = 20000000
# max_load_data = 65536
weight_decay_rate = 0.0
model_type = "MLP"  # MLP, MLP_Residual, MLP_Tanh, RNN, GRU, LSTM
is_used_loccode = True
is_used_Data_norm = True
model_name = 'MLP_e_4_tiran_80.5_test_81.6_val_85.7_is_121_hs_512_nh_20_bs_65536_lr_0.002_wd_0.0_ds_20000000_model_weights.bin'
index_name_list = ["sh000001", "sz399808"]

def pre_process(get_roll_yield_bar_df):
    exponential_list = []
    label = -1
    for i in range(len(get_roll_yield_bar_df)):
        nowday = datetime.datetime.strptime(get_roll_yield_bar_df.iloc[i][0], '%Y-%m-%d %H:%M:%S')
        if nowday.month == today.month and nowday.day == today.day:
            if (i + 119) <= len(get_roll_yield_bar_df):
                nowday_noon = datetime.datetime.strptime(get_roll_yield_bar_df.iloc[i + 119][0], '%Y-%m-%d %H:%M:%S')
                # print(nowday_noon)
                if nowday.hour == 9 and nowday.minute == 31 and nowday_noon.hour == 11 and nowday_noon.minute == 30:
                    exponential_list.extend(get_roll_yield_bar_df.iloc[i - 1:i + 120, 4])
                    if (i + 238) <= len(get_roll_yield_bar_df):
                        nowday_pm = datetime.datetime.strptime(get_roll_yield_bar_df.iloc[i + 238][0], '%Y-%m-%d %H:%M:%S')
                        # print(nowday_pm)
                        if nowday_pm.hour == 15 and nowday_pm.minute == 0:
                            if float(exponential_list[0]) > float(get_roll_yield_bar_df.iloc[i + 238][4]):
                                label = 1
                            else:
                                label = 0
            break
    # print(f"len {len(exponential_list)} get_roll_yield_bar_df {exponential_list}")
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
    # today = today - datetime.timedelta(days=2)
    if isTradeDay(today):
        msg = msg + str(today) + "\n"
        model = my_model.init_model(model_type, input_size, hidden_size, output_size, num_hidden_layers, device)

        model.load_state_dict(torch.load('./model/' + model_name))

        for index_name in index_name_list:
            # print(index_name)
            get_roll_yield_bar_df = ak.stock_zh_a_minute(symbol=index_name, period="1")
            # print(get_roll_yield_bar_df)
            is_valid, exponential_tensor, exponential_list, label = pre_process(get_roll_yield_bar_df)
            if is_valid:
                # print(f"len {len(exponential_tensor)} exponential_tensor {exponential_tensor}")
                outputs = model(exponential_tensor)
                result = torch.softmax(outputs, dim=0)
                _, predicted = torch.max(outputs, 0)
                # print(f"outputs {outputs} predicted {predicted} label {label}")
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



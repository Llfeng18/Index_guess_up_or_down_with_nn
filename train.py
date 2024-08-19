from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.optim.lr_scheduler as lr_scheduler
from transformers import AdamW, get_scheduler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import tensorflow as tf
import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
import cProfile
import random
import torch
import time
import json
import math
import csv
import os
import re
import model as my_model
import data_process as my_data_process
from data_process import MyDatasetLoader

pd.set_option('display.max_rows', None)      # 设置行不限制数量
pd.set_option('display.max_columns', None)   # 设置列不限制数量
pd.set_option('max_colwidth', 300)          # 设置value的显示长度为300
pd.set_option('display.width', 1000)        # 设置显示宽度，防止自动换行

torch.set_float32_matmul_precision('high')
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cpu")
print(f"using device: {device}")

total_batch_size = 65536
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
Data_norm_rate = 100000000.0
dummy_class = 0
load_model_name = ""
best_test_prob = 0.8

pre_process_file_dir = "D:/data/stock_market/Index/"
train_file = "pre_process_train"
test_file = "pre_process_test"
val_file = "pre_process_val"

use_compile = False

if model_type == "Transformer":
    is_used_loccode = False
if dummy_class:
    output_size = output_size + dummy_class
assert total_batch_size % batch_size == 0
assert hidden_size % n_head == 0
accumulation_steps = total_batch_size / batch_size

def train_loop(num_epochs, loader):
    global best_test_prob, train_num_epochs
    loss_list = []
    correct_list = []

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        batch_index = 0
        starttime = datetime.datetime.now()
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        norm = 0
        # for data, labels in train_loader:
            # data, labels = data.to(device), labels.to(device)
        prefetcher = my_data_process.data_prefetcher(loader)
        data, labels = prefetcher.next()
        while data is not None:
            # 前向传播
            with torch.no_grad():
                data = Data_norm_rate * (data / data[:, 0].unsqueeze(1) - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(data)
                outputs = outputs[:, :2]
                loss = torch.nn.functional.cross_entropy(outputs, labels) / accumulation_steps
            with torch.no_grad():
                batch_index = batch_index + 1
                _, predicted = torch.max(outputs[:, :2], 1)
                predicted_label = torch.nn.functional.one_hot(predicted).to(device)
                total += labels.size(0)
                correct += (predicted_label == labels).sum().item() / 2
            loss.backward()
            if batch_index % accumulation_steps == 0 or batch_index == len(loader):
                norm += torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # lr_scheduler.step()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.detach()
            data, labels = prefetcher.next()
        tiran_prob = correct / total
        test_correct, test_total = eval_model(test_loader, model, False)[0:2]
        val_correct, val_total = eval_model(val_loader, model, False)[0:2]
        test_prob = (test_correct + val_correct) / (test_total + val_total)
        val_prob = val_correct / val_total
        if test_prob > best_test_prob or epoch == (train_num_epochs - 1):
            torch.save(
                model.state_dict(),
                f'./train_model/{model_type}_e_{epoch + 1}_tiran_{tiran_prob*100:0.2f}_test_{test_prob*100:0.2f}_'
                f'val_{val_prob*100:0.2f}_bs_{batch_size}_as_{accumulation_steps}_is_{input_size}_hs_{hidden_size}'
                f'_nh_{n_head}_os_{output_size}_nhl_{num_hidden_layers}_'
                f'lr_{learning_rate}_wd_{weight_decay_rate}_ds_{max_load_data}_model_weights.bin'
            )
        if test_prob > best_test_prob:
            best_test_prob = test_prob
        endtime = datetime.datetime.now()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()/(len(loader)):.4f}, '
              f'norm: {norm / len(loader):.4f} , train:{(100 * tiran_prob):.2f}% , '
              f'test:{(100 * test_prob):.2f}% , val:{(100 * val_prob):.2f}% , time{(endtime - starttime)}')

    return loss_list, correct_list

def eval_model(loader, model, is_return_result_list):
    model.eval()
    true_result_list = []
    false_result_list = []
    false_detail_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        if not is_return_result_list:
        # for data, labels in loader:
        #     data, labels = data.to(device), labels.to(device)
            prefetcher = my_data_process.data_prefetcher(loader)
            data, labels = prefetcher.next()
            while data is not None:
                with torch.no_grad():
                    data = Data_norm_rate * (data / data[:, 0].unsqueeze(1) - 1)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    outputs = model(data)
                outputs = outputs[:, :2]
                _, predicted = torch.max(outputs, 1)
                predicted_label = torch.nn.functional.one_hot(predicted).to(device)
                total += labels.size(0)
                correct += (predicted_label == labels).sum().item() / 2
                data, labels = prefetcher.next()
        else:
            for data, labels, index, day in loader:
                data, labels = data.to(device), labels.to(device)
                with torch.no_grad():
                    data = Data_norm_rate * (data / data[:, 0].unsqueeze(1) - 1)
                outputs = model(data)
                outputs = outputs[:, :2]
                _, predicted = torch.max(outputs, 1)
                predicted_label = torch.nn.functional.one_hot(predicted).to(device)
                total += labels.size(0)
                correct += (predicted_label == labels).sum().item() / 2
                for i in range(len(predicted)):
                    result = torch.softmax(outputs[i], dim=0)
                    if (predicted_label[i] == labels[i]).sum().item() == 2:
                        true_result_list.append(result[predicted[i]] * 100)
                    else:
                        false_detail = []
                        false_detail.append(index[i])
                        false_detail.append(str(day[i][0].item()))
                        false_detail.append(str(day[i][1].item()))
                        false_detail.append(str(day[i][2].item()))
                        false_detail_list.append(false_detail)
                        false_result_list.append(result[predicted[i]] * 100)
    return correct, total, true_result_list, false_result_list, false_detail_list
    

if __name__ == '__main__':
    train_loader, test_loader, val_loader = my_data_process.get_loader_all(pre_process_file_dir, train_file,
        test_file, val_file, max_load_data, batch_size, False)

    model = my_model.init_model(model_type, input_size, hidden_size, output_size, num_hidden_layers, n_head, device)
    print(f"number of parameters: {sum(p.numel() for p in model.parameters()) / 1.0e6}M")
    if load_model_name != "":
        model.load_state_dict(torch.load('./model/' + load_model_name))
    if use_compile:
        model = torch.compile(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    test_correct, test_total = eval_model(test_loader, model, False)[0:2]
    val_correct, val_total = eval_model(val_loader, model, False)[0:2]
    test_prob = (test_correct + val_correct) / (test_total + val_total)
    val_prob = val_correct / val_total
    print(f'initial test:{(100 * test_prob):.2f}% , val:{(100 * val_prob):.2f}% , T_max:{math.ceil(train_num_epochs * len(train_loader) / accumulation_steps)}')
    
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=math.ceil(train_num_epochs * len(train_loader) / accumulation_steps),  # T_max 是一个完整周期的训练步骤数
        eta_min=0  # 可选，最小学习率，默认值为0
    )
    loss_list, correct_list = train_loop(train_num_epochs, train_loader)



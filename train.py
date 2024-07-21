from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm.auto import tqdm
import tensorflow as tf
import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
import random
import torch
import json
import csv
import os
import model as my_model
import data_process as my_data_process
import torch.optim.lr_scheduler as lr_scheduler
import cProfile
import re

pd.set_option('display.max_rows', None)      # 设置行不限制数量
pd.set_option('display.max_columns', None)   # 设置列不限制数量
pd.set_option('max_colwidth', 300)          # 设置value的显示长度为300
pd.set_option('display.width', 1000)        # 设置显示宽度，防止自动换行

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 65536
input_size = 121
hidden_size = 512
output_size = 2
num_hidden_layers = 20
learning_rate = 2e-3
num_epochs = 100
max_load_data = 20000000
max_load_data = 65537
weight_decay_rate = 0.0
model_type = "MLP"  # MLP, MLP_Residual, MLP_Tanh, RNN, GRU, LSTM
is_used_loccode = True
is_used_Data_norm = True

def train_loop():
    loss_list = []
    correct_list = []
    best_test_prob = 0.80
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        starttime = datetime.datetime.now()
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        # for data, labels in train_loader:
            # data, labels = data.to(device), labels.to(device)
        prefetcher = my_data_process.data_prefetcher(train_loader)
        data, labels = prefetcher.next()
        while data is not None:
            # 前向传播
            outputs = model(data)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                predicted_label = torch.nn.functional.one_hot(predicted).to(device)
                total += labels.size(0)
                correct += (predicted_label == labels).sum().item() / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            data, labels = prefetcher.next()
        tiran_prob = correct / total
        test_correct, test_total = eval_model(test_loader, model, False)[0:2]
        val_correct, val_total = eval_model(val_loader, model, False)[0:2]
        test_prob = (test_correct + val_correct) / (test_total + val_total)
        val_prob = val_correct / val_total
        if test_prob > best_test_prob or epoch == (num_epochs - 1):
            torch.save(
                model.state_dict(),
                f'./train_model/{model_type}_e_{epoch + 1}_tiran_{tiran_prob*100:0.1f}_test_{test_prob*100:0.1f}_val_{val_prob*100:0.1f}_is_{input_size}_hs_{hidden_size}_nh_{num_hidden_layers}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay_rate}_ds_{max_load_data}_model_weights.bin'
            )
        if test_prob > best_test_prob:
            best_test_prob = test_prob

        endtime = datetime.datetime.now()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, train:{(100 * tiran_prob):.2f}% , test:{(100 * test_prob):.2f}% , val:{(100 * val_prob):.2f}% , time{(endtime - starttime)}')
    print("Training completed.")

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
                # data, labels = data.to(device), labels.to(device)
            prefetcher = my_data_process.data_prefetcher(loader)
            data, labels = prefetcher.next()
            while data is not None:
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                predicted_label = torch.nn.functional.one_hot(predicted).to(device)
                total += labels.size(0)
                correct += (predicted_label == labels).sum().item() / 2
                data, labels = prefetcher.next()
        else:
            for data, labels, index, day in loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
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
                # print(f'{i} {predicted_label[i]} {labels[i]} {result[predicted[i]]*100} {result*100}')
    # print(f'Accuracy on validation set: {(100 * correct / total):.2f}%')
    return correct, total, true_result_list, false_result_list, false_detail_list
    

if __name__ == '__main__':
    train_loader, test_loader, val_loader = my_data_process.get_loader(max_load_data, is_used_Data_norm,
                                                                       is_used_loccode, batch_size, False)

    model = my_model.init_model(model_type, input_size, hidden_size, output_size, num_hidden_layers, device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs * len(train_loader),  # T_max 是一个完整周期的训练步骤数
        eta_min=0  # 可选，最小学习率，默认值为0
    )
    test_correct, test_total = eval_model(test_loader, model, False)[0:2]
    val_correct, val_total = eval_model(val_loader, model, False)[0:2]
    test_prob = (test_correct + val_correct) / (test_total + val_total)
    val_prob = val_correct / val_total
    print(f'initial test:{(100 * test_prob):.2f}% , val:{(100 * val_prob):.2f}%')
    train_loop()
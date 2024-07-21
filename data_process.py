from multiprocessing import Pool, cpu_count, Manager, freeze_support
from torch.utils.data import Dataset, DataLoader, random_split
from functools import partial
from time import sleep
from glob import glob
import pandas as pd
import numpy as np
import datetime
import shutil
import json
import lxml
import time
import torch
import csv
import os
import gc

class MyDatasetLoader_Tset(Dataset):
    def __init__(self, num_samples, input_data):
        self.num_samples = num_samples
        self.data = input_data[:, :121].astype(np.float32)
        self.labels = input_data[:, 125:127].astype(np.float32)
        self.index = input_data[:, 121].astype('U10')
        self.day = input_data[:, 122:125].astype(np.int8)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])
        index_name = self.index[idx]
        index_day = self.day[idx]
        return sample, label, index_name, index_day

class MyDatasetLoader(Dataset):
    def __init__(self, num_samples, input_data):
        self.num_samples = num_samples
        self.data = input_data[:, :121].astype(np.float32)
        self.labels = input_data[:, 125:127].astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])
        return sample, label

def load_data(data_dir, max_load_data, is_used_Data_norm, is_used_loccode):
    loaded_list = []
    with open(data_dir, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= max_load_data:
                break
            if not (np.isnan(np.array(row[0:121], dtype=np.float64)).any() or
                    int(float(row[0])) == 0):
                tmp_row = row[0:121] + row[241:247]
                loaded_list.append(tmp_row)

    for i in range(len(loaded_list)):
        if is_used_Data_norm:
            base = float(loaded_list[i][0])
            for j in range(0, 121):
                index = float(loaded_list[i][j])
                if is_used_loccode:
                    loaded_list[i][j] = str(int((index - base) * 1000000 / base + j * 10))
                else:
                    loaded_list[i][j] = str(int((index - base) * 1000000 / base))

    return loaded_list

class data_prefetcher():
    def __init__(self, loader):
        # loader 1：real
        # loader 2：fake
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def get_loader(max_load_data, is_used_Data_norm, is_used_loccode, batch_size, is_Test):
    pre_process_file_dir = "D:/data/stock_market/Index/"
    train_file = "pre_process_train.csv"
    test_file = "pre_process_test.csv"
    val_file = "pre_process_val.csv"

    test_loaded_list = load_data(pre_process_file_dir + test_file, max_load_data, is_used_Data_norm, is_used_loccode)
    if is_Test:
        test_dataset = MyDatasetLoader_Tset(len(test_loaded_list), np.array(test_loaded_list))
    else:
        test_dataset = MyDatasetLoader(len(test_loaded_list), np.array(test_loaded_list))
    del test_loaded_list

    val_loaded_list = load_data(pre_process_file_dir + val_file, max_load_data, is_used_Data_norm, is_used_loccode)
    if is_Test:
        val_dataset = MyDatasetLoader_Tset(len(val_loaded_list), np.array(val_loaded_list))
    else:
        val_dataset = MyDatasetLoader(len(val_loaded_list), np.array(val_loaded_list))
    del val_loaded_list

    train_loaded_list = load_data(pre_process_file_dir + train_file, max_load_data, is_used_Data_norm, is_used_loccode)
    if is_Test:
        train_dataset = MyDatasetLoader_Tset(len(train_loaded_list), np.array(train_loaded_list))
    else:
        train_dataset = MyDatasetLoader(len(train_loaded_list), np.array(train_loaded_list))
    del train_loaded_list
    gc.collect()

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    print("load_data success")

    return train_loader, test_loader, val_loader

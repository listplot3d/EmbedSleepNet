import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

"""
dataset.py中的关键设计：

    1.动态批处理（BatchedSleepDataset）：
       * 问题根源：不同受试者记录时长不同：以30秒为一个epoch，个体睡眠数据的epoch数量不同）
       * 解决方案：通过minibatch_size参数控制每个样本包含的epoch数量(连续睡眠片段)
       * 实现机制：
        - reshuffle()方法为每个受试者生成随机偏移量（shift）
        - 通过(start_idx:start_idx + minibatch_size)截取连续片段
        - 优势：保持时序连续性同时实现批规范化

    2.随机位移（reshuffle）：
        * max_skip = 5 * minibatch_size + subj_data.shape[0] % minibatch_size cur_skip = random.randint(0, max_skip)
        * 作用：每个epoch重新随机选择数据起始点
        * 效果：防止模型记忆固定数据顺序，增强泛化能力
    
    3.维度扩展（unsqueeze）：
        * 原始数据维度：[minibatch_size, 3000]（3000=30秒×100Hz）
        * unsqueeze(1)后：[minibatch_size, 1, 3000]
        * 原因：适配PyTorch Conv1d层的输入要求（通道维度在前）
"""

def load_sleep_data(data_path='preprocessed'):
    data = []
    labels = []
    for file in os.listdir(data_path):
        if not file.endswith('.npz'):
            continue
        path = os.path.join(data_path, file)
        loaded = np.load(path)
        data.append(loaded['x'])
        labels.append(loaded['y'])
    return data, labels


class BatchedSleepDataset(Dataset):
    def __init__(self, data, labels, minibatch_size=20):
        """动态批处理数据集
        
        参数：
            data: 多受试者的EEG信号列表 [n_subj][n_epochs, 3000]
            labels: 对应睡眠阶段标签 [n_subj][n_epochs]
            minibatch_size: 每个样本包含的连续睡眠时段数（默认30秒×20=10分钟）
        
        设计特点：
        1. 随机位移：每个epoch重置起始点防止过拟合
        2. 连续采样：保持睡眠阶段的时序连续性
        3. 动态长度：自动适配不同受试者的数据长度差异
        """
        self.data = data
        self.labels = labels
        self.minibatch_size = minibatch_size
        self.reshuffle()  # 初始化随机位移参数

    def reshuffle(self):
        
        self.shifts = []
        self.cur_lens = []
        self.total_len = 0
        for subj_data in self.data:
            max_skip = 5 * self.minibatch_size + subj_data.shape[0] % self.minibatch_size
            cur_skip = random.randint(0, max_skip)
            self.shifts.append(cur_skip)
            cur_len = (subj_data.shape[0] - cur_skip) // self.minibatch_size
            self.cur_lens.append(cur_len)
            self.total_len += cur_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        subj_idx = len(self.data) - 1
        for idx, subj_data in enumerate(self.data):
            cur_len = self.cur_lens[idx]
            if index >= cur_len:
                index -= cur_len
            else:
                subj_idx = idx
                break
        start_idx = self.shifts[subj_idx] + index * self.minibatch_size
        item_data = self.data[subj_idx][start_idx:start_idx + self.minibatch_size]
        item_labels = self.labels[subj_idx][start_idx:start_idx + self.minibatch_size]
        return torch.tensor(item_data, dtype=torch.float).unsqueeze(1), \
               torch.tensor(item_labels, dtype=torch.long).unsqueeze(1)


class SleepDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float).unsqueeze(1), \
               torch.tensor(self.labels[index], dtype=torch.long).unsqueeze(1)


def load_split_sleep_dataset(data_path='preprocessed', minibatch_size=20):
    data, labels = load_sleep_data(data_path)
    train_size = round(len(data) - 8)
    data_train, labels_train = data[:train_size], labels[:train_size]
    data_test, labels_test = data[train_size:], labels[train_size:]
    return BatchedSleepDataset(data_train, labels_train, minibatch_size), SleepDataset(data_test, labels_test)

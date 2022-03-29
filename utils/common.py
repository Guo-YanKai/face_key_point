#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 17:10
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : common.py
# @software: PyCharm
import numpy as np
from torch.utils.data import SubsetRandomSampler

def split_data_val(dataset, args, shuffle=True):
    """打乱数据，划分验证集
        参数：dataset：实例化后的Dataset对象
            args: 超参数
            shuffle:是否shuffle数据"""
    print("total sample:", len(dataset))
    valid_rate = args.valid_rate
    data_size = len(dataset)
    indices = list(range(data_size))  # 生成索引
    split = int(np.floor(valid_rate * data_size))  # np.floor返回不大于输入参数的最大整数
    if shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)  # 根据随机种子打散索引
    train_indices, val_indices = indices[split:], indices[:split]

    # 生成数据采样器和加载器
    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    print(f"train sample: {len(train_indices)},  val sample: {len(val_indices)}")
    return train_sample, val_sample
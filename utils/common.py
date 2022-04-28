#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 17:10
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : common.py
# @software: PyCharm
import numpy as np
from torch.utils.data import SubsetRandomSampler
from scipy import ndimage
from PIL import Image

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


def np_max_yx(arr):
    """
    :param arr: 二维数粗
    :return: 求一个数组中的最大坐标【y,x】，最大值：max_val
    """
    argmax_0 = np.argmax(arr, axis=0)
    max_0 = arr[argmax_0, np.arange(arr.shape[1])]
    argmax_1 = np.argmax(max_0)
    max_yx_pos = np.array([argmax_0[argmax_1], argmax_1])
    max_val = arr[max_yx_pos[0], max_yx_pos[1]]
    return max_val, max_yx_pos


def get_max_heatmap_activation(tensor, gauss_sigma):
    array = tensor.cpu().detach().numpy()

    activations = ndimage.gaussian_filter(array, sigma=gauss_sigma, truncate=1.0)
    max_val, max_pos = np_max_yx(activations)
    return max_val, max_pos


def get_predicted_landmarks(pred_heatmaps, ori_img_path, gauss_sigma):
    n_landmarks = pred_heatmaps.shape[0]
    heatmap_y, heatmap_x = pred_heatmaps.shape[1:]
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    ori_img = Image.open(ori_img_path).convert("RGB")

    ori_img_x = ori_img.size[0]
    ori_img_y = ori_img.size[1]
    rescale = np.array([ori_img_y, ori_img_x]) / np.array([heatmap_y, heatmap_x])
    for i in range(n_landmarks):
        max_activation, pred_yx = get_max_heatmap_activation(pred_heatmaps[i], gauss_sigma)
        pred_yx = np.around(pred_yx * rescale)
        pred_landmarks[i] = pred_yx
        max_activations[i] = max_activation
    return pred_landmarks, max_activations
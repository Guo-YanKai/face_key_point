#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 17:50
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : metric.py
# @software: PyCharm

import torch
import numpy as np
import torch.nn.functional as F

class LossAverage(object):
    """计算并存储当前损失值和平均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
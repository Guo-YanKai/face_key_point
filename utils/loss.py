#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 17:35
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : loss.py
# @software: PyCharm
from torch import nn
import numpy as np
from collections import Counter
import torch
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        assert pred.size() == label.size()
        # a = label != 0
        # print("a:", a.shape)
        # print(Counter(a[0].detach().cpu().numpy().ravel()))
        # a = self.alpha * (label != 0).float()
        # print("a:",a)
        # b = ((torch.ones_like(pred)-pred)**self.gamma)
        # print("b:", b)
        # c = -(a*b)
        # print("c:", c.mean())

        tisone = -(self.alpha * (label != 0).float()) * ((torch.ones_like(pred) - pred) ** self.gamma) * torch.log(pred)
        # print("tisone:", tisone)
        tiszero = -((1 - self.alpha) * (label == 0).float()) * (pred ** self.gamma) * torch.log(
            torch.ones_like(pred) - pred)
        loss = tisone + tiszero
        print("loss:", loss.mean())

        return 1



class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))



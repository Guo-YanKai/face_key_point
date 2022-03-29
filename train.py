#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 18:35
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : train.py.py
# @software: PyCharm

from config import args
import torch
from models.unet import unet
from dataset.landmark import LandMarkDataset


def train(train_loader):
    return 0




if  __name__ == "__main__":
    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    print(device)

    dataset = LandMarkDataset(args)


    net = unet(3, args.n_landmark)
    print(net)
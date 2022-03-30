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
from utils.common import split_data_val
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.logger import Train_Logger
from utils.metric import LossAverage
import os
from tqdm import tqdm
from collections import OrderedDict
from collections import Counter
from utils.loss import FocalLoss, WingLoss, AdaptiveWingLoss

import numpy as np

def val(net, val_loader, criterion, device, args):
    net.eval()
    val_loss = LossAverage()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch_images = batch["image"].to(device, dtype=torch.float32)
            batch_masks = batch["label"].to(device, dtype=torch.float32)
            output = net(batch_images)
            loss = criterion(output, batch_masks)
            val_loss.update(loss.item(), batch_images.size(0))
            val_log = OrderedDict({"Val_Loss": val_loss.avg})
    return val_log


def train(net, train_loader, criterion, optimizer, device, args):
    print("=====Epoch:{}======lr:{}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
    net.train()
    train_loss = LossAverage()
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        batch_images = batch["image"].to(device, dtype=torch.float32)
        batch_labels = batch["label"].to(device, dtype=torch.float32)

        output = net(batch_images)
        # print("batch_labels:", batch_labels.shape)
        # print("output:", output.shape)
        # print("batch_labels[0]:", Counter(np.array(batch_labels[0].detach().cpu()).ravel()))
        loss = criterion(output, batch_labels)

        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), batch_images.size(0))

    train_log = OrderedDict({"Train_Loss": train_loss.avg})
    train_log.update({"lr": optimizer.state_dict()["param_groups"][0]["lr"]})
    return train_log



if __name__ == "__main__":
    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")

    dataset = LandMarkDataset(args)

    train_sample, val_sample = split_data_val(dataset, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              num_workers=args.n_threads, sampler=train_sample, pin_memory=False)
    val_loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.n_threads, sampler=val_sample, pin_memory=False)

    net = unet(3, args.n_landmark).to(device)


    if args.loss == "CRE":
        criterion = nn.CrossEntropyLoss()
    elif args.loss=="FocalLoss":
        criterion = FocalLoss()
    elif args.loss=="WingLoss":
        criterion = WingLoss()
    elif args.loss =="AdaptiveWingLoss":
        criterion = AdaptiveWingLoss()
    else:
        criterion = nn.MSELoss()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.9)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.9)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    log_save_path = os.path.join(args.save_path, args.net_name, args.loss, args.optimizer)
    os.makedirs(log_save_path, exist_ok=True)

    log = Train_Logger(log_save_path, "train_log")
    best = [0, float("inf")]
    trigger = 0
    for epoch in range(1, args.epochs + 1):
        train_log = train(net, train_loader, criterion, optimizer, device, args)

        val_log = val(net, val_loader, criterion, device, args)
        scheduler.step()
        log.update(epoch, train_log, val_log)
        # save checkpoints
        state = {"net": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch}

        torch.save(state, os.path.join(log_save_path, "latest_model.pth"))
        trigger += 1

        if val_log["Val_Loss"] < best[-1]:
            print("save best model")
            torch.save(state, os.path.join(log_save_path, "best_model.pth"))
            best[0] = epoch
            best[1] = val_log["Val_Loss"]
            trigger = 0
        print("Best Performance at Epoch:{}|{}".format(best[0], best[1]))
        # 早停
        if trigger >= args.early_stop:
            print("=>early stopping")
            break
    torch.cuda.empty_cache()

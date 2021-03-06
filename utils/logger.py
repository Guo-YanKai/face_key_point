#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 17:41
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : logger.py
# @software: PyCharm

from collections import OrderedDict
import numpy as np
import torch, random
from tensorboardX import SummaryWriter
import pandas as pd
import os


def dict_round(dic, num):
    """将dic中的值取num位小数"""
    for k, v in dic.items():
        if k == "lr":
            dic[k] = round(v, num * 2)
        else:
            dic[k] = round(v, num)
    return dic


class Train_Logger():
    """保存训练过程中的各种指标，csv保存、tensorboard可视化"""

    def __init__(self, save_path, save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self, epoch, train_log, val_log):
        # 有序字典
        item = OrderedDict({"epoch": epoch})
        item.update(train_log)
        item.update(val_log)
        item = dict_round(item, 4)
        print("\033[0:32m Train: \033[0m", train_log)
        print("\033[0:32m Val: \033[0m", val_log)
        self.updata_csv(item)
        self.updata_tensorboard(item)

    def updata_csv(self, item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = self.log.append(item, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv("%s/%s.csv" % (self.save_path, self.save_name), index=False)

    def updata_tensorboard(self, item):
        if self.summary is None:
            self.summary = SummaryWriter("%s/" % (self.save_path))
        epoch = item["epoch"]
        for key, value in item.items():
            if key != "epoch":
                self.summary.add_scalar(key, value, epoch)


if __name__ =="__main__":
    save_path = r"D:\code\work_code\github_code\face_key_point"
    save_name ="train"
    log = Train_Logger(save_path, save_name)
    train_log = OrderedDict({"train_loss": 0.55964})
    train_log.update({"lr": 0.0001})
    val_log = OrderedDict({"val_loss": 0.65})

    print(train_log)
    log.update(1,train_log, val_log)


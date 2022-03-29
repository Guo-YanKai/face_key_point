#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 11:59
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : config.py.py
# @software: PyCharm

import argparse

parser = argparse.ArgumentParser("hyper-parameters management")

# 硬件选项
parser.add_argument("--n_threads", type=int, default=0, help="number of threads")
parser.add_argument("--device", default="0", help="use gpu only")
parser.add_argument("--seed", type=int, default=202203, help="random seed")

# 各种文件存储路径
parser.add_argument("--ori_images", default=r"D:\code\work_code\github_code\face_key_point\train_data\face_image", help="origin images path")
parser.add_argument("--annotations_path", default=r"D:\code\work_code\github_code\face_key_point\train_data\face_label", help="origin label txt path")
parser.add_argument("--save_path", default="./experiments", help="save model path")

# 数据处理的参数
parser.add_argument("--n_landmark", type=int, default=6, help="number of classes")
parser.add_argument("--resize", type=int, default=224, help="origin image resize")
parser.add_argument("--gauss_amplitude", type=float, default=1000.0, help="高斯振幅")
parser.add_argument("--gauss_sigma", type =float , default=5.0, help="高斯滤波器sigama")
# parser.add_argument("--GAUSSIAN_TRUNCATE", type =float , default=1.0, help="高斯滤波器参数")

parser.add_argument("--valid_rate", type=float, default=0.2, help="验证集划分率")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")

args = parser.parse_args()

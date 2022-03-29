#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 11:56
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : landmark.py
# @software: PyCharm

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy import ndimage
class LandMarkDataset(Dataset):
    def __init__(self, args):
        super(LandMarkDataset, self).__init__()
        self.ori_images = args.ori_images
        self.annotations_path = args.annotations_path
        self.resize = args.resize
        self.gauss_amplitude = args.gauss_amplitude
        self.gauss_sigma = args.gauss_sigma
        self.image_names = os.listdir(self.ori_images)
        self.transforme = transforms.Compose([transforms.Resize((self.resize, self.resize)),
                                              transforms.ToTensor()
                                              ])

    def __getitem__(self, item):
        ori_image = Image.open(os.path.join(self.ori_images, self.image_names[item])).convert("RGB")
        ori_x, ori_y = ori_image.size

        image = self.transforme(ori_image)
        txt_name = str(self.image_names[item].split(".")[0]) + ".txt"

        annots_path = os.path.join(self.annotations_path, txt_name)
        annots = self.get_annots_for_image(annots_path, ori_x, ori_y)

        y = self.create_true_heatmaps(annots, self.resize, self.gauss_amplitude)
        for i in range(y.shape[0]):
            y[i]  = ndimage.gaussian_filter(y[i], sigma=self.gauss_sigma, truncate=1.0)

        y = torch.from_numpy(y).float()
        name = self.image_names[item].split(".")[0]
        return {"image":image, "label":y, "name":name}


    def get_annots_for_image(self, annots_path, ori_x,  ori_y):
        ori_annots = []
        with open(annots_path, "r") as f:
            for line in f.readlines():
                ori_annots.append(line.strip("\n"))
        ori_annots = [l.split(",") for l in ori_annots]
        ori_annots = [(float(l[0]), float(l[1])) for l in ori_annots]
        ori_annots = np.array(ori_annots)

        scale = np.array([args.resize, args.resize], dtype=float)/np.array([ori_x, ori_y], dtype=float)
        annots = np.round(ori_annots*scale).astype("int32")

        return annots

    @staticmethod
    def create_true_heatmaps(annots, image_size, amplitude):
        heatmap = np.zeros((annots.shape[0], image_size, image_size))
        for i, pos in enumerate(annots):
            x, y = pos
            heatmap[i, y, x] = amplitude
        return heatmap

    def __len__(self):
        return len(self.image_names)


from config import args

if __name__ == "__main__":
    train_ds = LandMarkDataset(args)
    print(len(train_ds))
    data = train_ds[8]
    print("image:",data["image"].shape, "label", data["label"].shape, "name:", data["name"])


    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    for batch  in train_dl:
        print(batch["image"].shape)
        print(batch["label"].shape)
        print(batch["name"])
        break
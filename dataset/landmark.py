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
from config import args
import cv2

class LandMarkDataset(Dataset):
    def __init__(self,  args):
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
            y[i] = ndimage.gaussian_filter(y[i], sigma=self.gauss_sigma, truncate=1.0)

        y = torch.from_numpy(y).float()
        name = self.image_names[item].split(".")[0]
        return {"image": image, "label": y, "name": name}

    def get_annots_for_image(self, annots_path, ori_x, ori_y):
        ori_annots = []
        with open(annots_path, "r") as f:
            for line in f.readlines():
                ori_annots.append(line.strip("\n"))
        ori_annots = [l.split(",") for l in ori_annots]
        ori_annots = [(float(l[0]), float(l[1])) for l in ori_annots]
        ori_annots = np.array(ori_annots)

        scale = np.array([args.resize, args.resize], dtype=float) / np.array([ori_x, ori_y], dtype=float)
        annots = np.round(ori_annots * scale).astype("int32")

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


class TestDataset(Dataset):
    def __init__(self, args):
        super(TestDataset, self).__init__()
        self.test_ori_image = args.test_data
        self.names = os.listdir(self.test_ori_image)
        self.resize = args.resize
        self.transforme = transforms.Compose([transforms.Resize((self.resize, self.resize)),
                                              transforms.ToTensor()
                                              ])

    def __getitem__(self, item):
        ori_image = Image.open(os.path.join(self.test_ori_image, self.names[item])).convert("RGB")
        ori_x, ori_y = ori_image.size

        image = self.transforme(ori_image)
        name = self.names[item]
        return {"image": image, "name": name}

    def __len__(self):
        return len(self.names)





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
    print("array:", array.shape)
    activations = ndimage.gaussian_filter(array, sigma=gauss_sigma, truncate=1.0)
    max_val, max_pos = np_max_yx(activations)
    return max_val, max_pos


def get_predicted_landmarks(pred_heatmaps, ori_img_path, gauss_sigma):
    n_landmarks = pred_heatmaps.shape[0]
    print("n_landmarks:", n_landmarks)
    heatmap_y, heatmap_x = pred_heatmaps.shape[1:]
    print("heatmap_y:", heatmap_y, "heatmap_x:", heatmap_x)
    pred_landmarks = np.zeros((n_landmarks, 2))
    max_activations = np.zeros(n_landmarks)
    ori_img = Image.open(ori_img_path).convert("RGB")
    print(ori_img.size)
    ori_img_x = ori_img.size[0]
    ori_img_y = ori_img.size[1]
    rescale = np.array([ori_img_y, ori_img_x]) / np.array([heatmap_y, heatmap_x])
    print(rescale)

    for i in range(n_landmarks):
        max_activation, pred_yx = get_max_heatmap_activation(pred_heatmaps[i], gauss_sigma)
        pred_yx = np.around(pred_yx * rescale)
        pred_landmarks[i] = pred_yx
        max_activations[i] = max_activation
    return pred_landmarks, max_activations




if __name__ == "__main__":
    test_ds = TestDataset(args)
    print(len(test_ds))
    print(test_ds.__getitem__(4))

    # train_ds = LandMarkDataset(args)
    # print(len(train_ds))
    # # data = train_ds[8]
    # # print("image:",data["image"].shape, "label", data["label"].shape, "name:", data["name"])
    #
    # train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # for batch in train_dl:
    #     image = batch["image"]
    #     heatmap = batch["label"]
    #     names = batch["name"]
    #     break
    # print(names)
    # for i in range(args.batch_size):
    #     i = 1
    #     ori_image_path = os.path.join(args.ori_images, names[i] + ".png")
    #     print("pred_heatmaps:", heatmap[i].shape)
    #     print("ori_image_path：", ori_image_path)
    #     pred_landmarks, max_activations = get_predicted_landmarks(pred_heatmaps=heatmap[i],
    #                                                               ori_img_path=ori_image_path,
    #                                                               gauss_sigma=args.gauss_sigma)
    #     print("pred_landmarks：", type(pred_landmarks))
    #     ori_image = cv2.imread(ori_image_path)
    #     print(ori_image.shape)
    #     for j in range(len(pred_landmarks)):
    #         x = int(pred_landmarks[j][1])
    #         y = int(pred_landmarks[j][0])
    #         cv2.circle(ori_image, (x, y), radius=1, color=[0, 0, 255], thickness=4)
    #         cv2.putText(ori_image, text=str(j + 1), org=(x, y),
    #                     fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=[0, 255, 0], thickness=2)
    #     cv2.imwrite(f"{names[i]}_labeld.png", ori_image)
    #     break

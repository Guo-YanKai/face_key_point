#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 18:36
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : test.py.py
# @software: PyCharm
from config import args
from dataset.landmark import TestDataset
import torch
from torch.utils.data import DataLoader
from models.unet import unet
import os
from tqdm import tqdm
from utils.common import get_predicted_landmarks, get_max_heatmap_activation, np_max_yx
import cv2
import numpy as np
import pandas as pd
import csv

def label(out_csv_path,  args):
    output_labeled_path =  os.path.join(args.test_output, "labeled2")
    os.makedirs(output_labeled_path, exist_ok=True)
    csv_content = csv.reader(open(out_csv_path))
    rows = [row for row in csv_content]
    for i in range(1, len(rows)):
        row = rows[i]
        name = row[1]
        ori_image_path = os.path.join(args.test_data, name)
        img = cv2.imread(ori_image_path)

        for dot_index in range(0, args.n_landmark):
            x = int(row[args.n_landmark+8 + dot_index].replace('.0', ''))
            y = int(row[args.n_landmark+2 + dot_index].replace('.0', ''))
            cv2.circle(img, (x, y), radius=1,
                       color=[0, 0, 255],
                       thickness=4)
            cv2.putText(img, text=str(dot_index + 1), org=(x, y),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=[0, 255, 0], thickness=2)
        save_image_path = os.path.join(output_labeled_path, name)
        cv2.imwrite(save_image_path, img)

def predict(test_loader, net, args):
    columns = ['file'] + [f'{i}_act' for i in range(args.n_landmark)] + \
              [f'{i}_y' for i in range(args.n_landmark)] + \
              [f'{i}_x' for i in range(args.n_landmark)]
    index = np.arange(len(test_loader))
    df = pd.DataFrame(columns=columns, index=index)

    net.eval()
    with torch.no_grad():
        n_processed = 0
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch_images = batch["image"].to(device, dtype=torch.float32)
            names = batch["name"]
            pred_heatmaps = net(batch_images)
            batch_size = pred_heatmaps.shape[0]
            for i in range(batch_size):
                ori_image_path = os.path.join(args.test_data, names[i])
                pred_landmarks, max_activations = get_predicted_landmarks(pred_heatmaps[i],
                                                                          ori_image_path,
                                                                          args.gauss_sigma)

                for j in range(pred_landmarks.shape[0]):
                    row = n_processed + i
                    df.iloc[row]["file"]=names[i]
                    df.iloc[row]["file"] = names[i]
                    df.iloc[row][f"{j}_act"] = max_activations[j]
                    df.iloc[row][f"{j}_y"] = pred_landmarks[j][0]
                    df.iloc[row][f"{j}_x"] = pred_landmarks[j][1]

            n_processed += batch_size


    return df


if __name__ == "__main__":
    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    print("device:", device)

    args.batch_size = 1

    test_data = TestDataset(args)


    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             num_workers=args.n_threads, pin_memory=False)

    net = unet(3, args.n_landmark).to(device)

    model_pth = os.path.join(args.save_path, args.net_name, args.loss, args.optimizer)
    print("model_pth:", model_pth)
    ckpt = torch.load("{}\\best_model.pth".format(model_pth), map_location=device)
    net.load_state_dict(ckpt["net"])
    print("Model loaded！")


    out_csv_path = args.test_output+"\predictions.csv"

    predictions = predict(test_loader, net, args)
    predictions.to_csv(out_csv_path)


    if args.label:
        label(out_csv_path, args)


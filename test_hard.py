#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 15:11
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : test_hard.py
# @software: PyCharm

import cv2
import  matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = r"D:\code\work_code\github_code\face_key_point\train_data\hard\1095a2c3891f4a77a8459c247c5eb3f4.JPG"
    txt_path = r"D:\code\work_code\github_code\face_key_point\train_data\hard\1000469P1.txt"
    img = cv2.imread(img_path)
    with open(txt_path, "r") as f:
        annotions = f.readlines()

    for i in range(len(annotions)):
        annotion = annotions[i].strip().split(",")
        print("annotion:",annotion)
        x, y = list(map(float, annotion))
        x,y = int(x), int(y)
        print(x,y)
        cv2.circle(img, (x, y), 4, (0, 255, 0), 2)
        cv2.putText(img, str(i), (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 4, [255, 0, 0], 2)
    plt.imshow(img)
    plt.show()


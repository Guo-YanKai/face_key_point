#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 11:55
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : unet.py
# @software: PyCharm

import torch
from torch import nn, optim
from torch.nn import functional as F
from models.weights_init import init_weights

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.4):
        super(DownBlock, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(p=drop) if drop != 0 else None
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.mp(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.4):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.drop = nn.Dropout2d(p=drop) if drop != 0 else None
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self,x, x_stored):
        x = self.up(x)
        x = torch.cat([x, x_stored], dim=1)
        if self.drop is not None:
            x = self.drop(x)
        x = self.conv(x)
        return x


class unet(nn.Module):
    def __init__(self, in_ch, out_ch, predict_gaussian=False):
        super(unet, self).__init__()
        self.drop = [0.4, 0.4, 0.4, 0.4]
        self.up_drop = [0.4, 0.4, 0.4, 0.4]
        self.inconv = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128, drop=self.drop[0])
        self.down2 = DownBlock(128, 256, drop=self.drop[1])
        self.down3 = DownBlock(256, 512, drop=self.drop[2])
        self.down4 = DownBlock(512, 1024, drop=self.drop[3])

        self.up1 = UpBlock(1024, 512, drop=self.up_drop[0])
        self.up2 = UpBlock(512,256, drop=self.up_drop[1])
        self.up3 = UpBlock(256, 128, drop=self.up_drop[2])
        self.up4 = UpBlock(128, 64, drop=self.up_drop[3])

        self.outconv = nn.Conv2d(64, out_ch, 1)
        self.predict_gaussian = predict_gaussian


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type="kaiming")
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type="kaiming")



    def forward(self,x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = x5
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.outconv(x)

        x_final = x
        if self.predict_gaussian:
            n_landmark = x.shape[1]//2
            x_final = x.clone()
            x_final[:,n_landmark:,:,:] = F.softplus(x[:,n_landmark:, :, :])+1e-6

        return x_final

from config import args
if __name__ =="__main__":
    x = torch.randn((2, 3, 224,224))
    net = unet(3, out_ch=args.n_landmark)
    print(net(x).shape)
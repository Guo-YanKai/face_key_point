#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 10:56
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : hourglass.py
# @software: PyCharm

import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.in_ch = in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=(kernel_size - 1) // 2)
        self.bn = None
        self.relu = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        assert x.size()[1] == self.in_ch, "{}{}".format(x.size()[1], self.in_ch)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Residual(nn.Module):
    """
    残差结构：只改变通道数，不改变输入的大小
    """

    def __init__(self, in_ch, out_ch):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.conv1 = Conv(in_ch, int(out_ch / 2), kernel_size=1, stride=1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_ch / 2))

        self.conv2 = Conv(int(out_ch / 2), int(out_ch / 2), kernel_size=3, stride=1, bn=False, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_ch / 2))

        self.conv3 = Conv(int(out_ch / 2), out_ch, kernel_size=1, stride=1, relu=False)

        self.skip_layer = Conv(in_ch, out_ch, kernel_size=1, stride=1, relu=False)  #
        if in_ch == out_ch:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        return out


# class Hourglass(nn.Module):
#     def __init__(self, n, f, bn=True, increase=0):
#         super(Hourglass, self).__init__()
#         nf = f + increase
#         self.up1 = Residual(f, f)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.low1 = Residual(f, nf)
#         self.n = n
#
#         if self.n > 1:
#             self.low2 = Hourglass(n - 1, nf, bn=bn)
#         else:
#             self.low2 = Residual(nf, nf)
#
#         self.low3 = Residual(nf, f)
#         self.up2 = nn.Upsample(scale_factor=2, mode="nearest", align_corners=True)
#
#     def forward(self, x):
#         up1 = self.up1(x)
#         pool1 = self.pool1(x)
#         low1 = self.low1(pool1)
#         low2 = self.low2(low1)
#         low3 = self.low3(low2)
#         up2 = self.up2(low3)
#
#         return up1 + up2


class PoseNet(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        # 输入[N,3,224,224]
        self.pre = nn.Sequential(
            Conv(in_ch=3, out_ch=64, kernel_size=7, stride=2, bn=True, relu=True),  # =>[N,64,112,112]
            Residual(in_ch=64, out_ch=128),  # 只改变通道数，=>[N,128,112,112]
            nn.MaxPool2d(kernel_size=2, stride=2),  # ==>[N,128,56,56]
            Residual(128, 128),  # ==>[N,128,56,56]
            Residual(128, in_ch)
        )

    def forward(self, x):
        return x


class HgResBlcok(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(HgResBlcok, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        midplanes = outplanes // 2

        self.bn_1 = nn.BatchNorm2d(inplanes)
        self.conv_1 = nn.Conv2d(inplanes, midplanes, kernel_size=1, stride=stride)
        self.bn_2 = nn.BatchNorm2d(midplanes)
        self.conv_2 = nn.Conv2d(midplanes, midplanes, kernel_size=3, stride=1, padding=1,)
        self.bn_3 = nn.BatchNorm2d(midplanes)
        self.conv_3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

        if inplanes != outplanes:
            self.conv_skip = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        out = self.bn_1(x)
        out = self.conv_1(out)
        out = self.relu(out)

        out = self.bn_2(out)
        out = self.conv_2(out)
        out = self.relu(out)

        out = self.bn_3(out)
        out = self.conv_3(out)
        out = self.relu(out)
        if self.inplanes != self.outplanes:
            residual = self.conv_skip(residual)
        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, depth, nFeat, nModules, resBlocks):
        """
        定义单个Hourglass Module:用到了递归
        :param depth: 递归的深度
        :param nFeat: 结构中的通道数
        :param nModules: 设定block的通道数
        :param resBlocks: 残差模块
        """
        super(Hourglass, self).__init__()

        self.depth = depth
        self.nFeat = nFeat
        self.nModules = nModules
        self.resBlocks = resBlocks

        self.hg = self._make_hourglass()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_residual(self, n):
        return nn.Sequential(*[self.resBlocks(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_hourglass(self):
        hg = []
        for i in range(self.depth):
            res = [self._make_residual(self.nModules) for _ in range(3)]
            if i == (self.depth - 1):
                res.append(self._make_residual(self.nModules))
            hg.append(nn.ModuleList(res))

        return nn.ModuleList(hg)

    def _hourglass_forward(self, deth_id, x):
        up1 = self.hg[deth_id][0](x)
        low_1 = self.downsample(x)
        low_1 = self.hg[deth_id][1](low_1)

        if deth_id == (self.depth - 1):
            low_2 = self.hg[deth_id][3](low_1)
        else:
            low_2 = self._hourglass_forward(deth_id + 1, low_1)

        low_3 = self.hg[deth_id][2](low_2)
        up2 = self.upsample(low_3)
        print("******************")
        print("up1:",up1.shape)
        print("up2:", up2.shape)
        return up1 + up2

    def forward(self, x):
        return self._hourglass_forward(0, x)


class HourglassNet(nn.Module):
    def __init__(self, nStacks, nModules, nFeat, nClasses, resBlock=HgResBlcok, inplanes=3):
        super(HourglassNet, self).__init__()
        self.nStacks = nStacks
        self.nModules = nModules
        self.nFeat = nFeat
        self.nClasses = nClasses
        self.resBlock = resBlock
        self.inplanes = inplanes

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []

        for i in range(nStacks):
            hg.append(Hourglass(depth=4, nFeat=nFeat, nModules=nModules, resBlocks=resBlock))
            res.append(self._make_residual(nModules))
            fc.append(self._make_fc(nFeat, nFeat))
            score.append(nn.Conv2d(nFeat, nClasses, kernel_size=1))
            if i<(nStacks-1):
                fc_.append(nn.Conv2d(nFeat, nFeat, kernel_size=1))
                score_.append(nn.Conv2d(nClasses, nFeat, kernel_size=1))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_head(self):
        self.conv_1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3)
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.res_1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res_2 = self.resBlock(128, 128)
        self.res_3 = self.resBlock(128, self.nFeat)


    def _make_residual(self,n):
        return nn.Sequential(*[self.resBlock(self.nFeat, self.nFeat) for _ in range(n)])

    def _make_fc(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        self._make_head()
        x = self.conv_1(x)
        x = self.bn_1(x)
        x= self.relu(x)

        x = self.res_1(x)
        x = self.pool(x)
        x = self.res_2(x)
        x = self.res_3(x)
        print(x.shape)
        out = []

        for i in range(self.nStacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)

            if i<(self.nStacks-1):
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out

if __name__ == "__main__":
    # https://blog.csdn.net/github_36923418/article/details/81030883
    net = Hourglass(depth=4, nFeat=256, nModules=1, resBlocks=HgResBlcok)
    x= torch.randn((2, 256, 112, 112))
    # net2 = HourglassNet(nStacks=2, nModules=1, nFeat=256, nClasses=6, resBlock=HgResBlcok, inplanes=3)
    print(net(x).shape)
    # x= torch.randn((2, 128, 14, 14))
    # a = nn.MaxPool2d(kernel_size=2, stride=2)
    # print(a(x).shape)



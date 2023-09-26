#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：PNet.py
@Author  ：huangxj
@Date    ：2023/9/7 10:55
该方法保存 PNet 的网络
'''

import torch
import torch.nn as nn
import torchvision
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1))
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1, 1))
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=136, kernel_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        return
        class_out: 对应像素点是否存在人脸
        bbox_out:  回归偏移矩阵
        landmark_out : 人脸关键点
        """
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        # 分类是否人脸的卷积输出层
        class_out = self.conv4_1(x)
        class_out = torch.squeeze(class_out, dim=2)
        class_out = torch.squeeze(class_out, dim=2)
        # 人脸box的回归卷积输出层
        bbox_out = self.conv4_2(x)
        bbox_out = torch.squeeze(bbox_out, dim=2)
        bbox_out = torch.squeeze(bbox_out, dim=2)
        # 5个关键点的回归卷积输出层
        landmark_out = self.conv4_3(x)
        landmark_out = torch.squeeze(landmark_out, dim=2)
        landmark_out = torch.squeeze(landmark_out, dim=2)
        return class_out, bbox_out, landmark_out


if __name__ == '__main__':
    # 生成（h,w,c)的图片
    image = np.ones((70, 70, 3), dtype=np.float32)
    print(image.shape)
    # 转成 (c,h,w)的tensor
    img = torchvision.transforms.ToTensor()(image)
    # 新增一个维度，变成（n,c,h,w)
    img = torch.unsqueeze(img, dim=0)
    net = PNet()
    class_out, bbox_out, landmark_out = net(img)
    print(class_out.shape)
    print(bbox_out.shape)
    print(landmark_out.shape)


#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：ONet.py
@Author  ：huangxj
@Date    ：2023/9/18 9:40
存放着不同的网络结构，用以替换ONet的网络结构
'''

import torch.nn as nn
import torch


class ONet(nn.Module):
    """
    经过多次测试，该网络对人脸框有良好的识别能力，对于68点人脸关键点则表现出很差的识别情况
    输入参数改成98 * 96 ，增加了网络的感受野最终得训练效果也很差
    """
    def __init__(self):
        super(ONet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.prelu4 = nn.PReLU()
        # 多维张量展平成一维张量
        self.flatten = nn.Flatten()
        # 全连接层
        self.fc = nn.Linear(in_features=128 * 9 * 9, out_features=256)
        self.class_fc = nn.Linear(in_features=256, out_features=2)
        self.bbox_fc = nn.Linear(in_features=256, out_features=4)
        self.landmark_fc = nn.Linear(in_features=256, out_features=68 * 2)

        # 使用nn.init.kaiming_normal方法进行参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))
        x = self.flatten(x)
        x = self.fc(x)
        # 分类是否人脸的卷积输出层
        class_out = self.class_fc(x)
        # 人脸box的回归卷积输出层
        bbox_out = self.bbox_fc(x)
        # 5个关键点的回归卷积输出层
        landmark_out = self.landmark_fc(x)
        return class_out, bbox_out, landmark_out





if __name__ == '__main__':
    import numpy as np
    import torchvision

    image = np.ones((96, 96, 3), dtype=np.float32)
    print(image.shape)
    # 转成 (c,h,w)的tensor
    img = torchvision.transforms.ToTensor()(image)
    # 新增一个维度，变成（n,c,h,w)
    img = torch.unsqueeze(img, dim=0)
    net = ONet()
    class_out, bbox_out, landmark_out = net(img)

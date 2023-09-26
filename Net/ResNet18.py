#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：ResNet18.py
@Author  ：huangxj
@Date    ：2023/9/20 11:36 
'''

import torch
import torch.nn as nn
import torchvision.models as models

# 定义ResNet-18作为特征提取器
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)  # 使用预训练的ResNet-18
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后两层（全局平均池化和全连接层）

    def forward(self, x):
        return self.features(x)

# 定义O-Net替代模型
class FaceDetectionNet(nn.Module):
    def __init__(self):
        super(FaceDetectionNet, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)  # 置信度输出
        self.fc3 = nn.Linear(256, 4)  # 边界框输出
        self.fc4 = nn.Linear(256, 68 * 2)  # 68个关键点输出

    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.conv(features)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        confidence = torch.sigmoid(self.fc2(x))
        bb_offset = self.fc3(x)
        landmark_offset = self.fc4(x)
        return confidence, bb_offset, landmark_offset


# 单独训练人脸68关键点的网络
class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        landmarks = self.model(x)
        return landmarks



if __name__ == '__main__':

    import torchvision
    import numpy as np
    # 创建FaceDetectionNet实例
    # 生成（h,w,c)的图片
    image = np.ones((96, 96, 3), dtype=np.float32)
    print(image.shape)
    # 转成 (c,h,w)的tensor
    img = torchvision.transforms.ToTensor()(image)
    # 新增一个维度，变成（n,c,h,w)
    img = torch.unsqueeze(img, dim=0)
    net  = FaceDetectionNet()
    class_out, bbox_out, landmark_out = net(img)


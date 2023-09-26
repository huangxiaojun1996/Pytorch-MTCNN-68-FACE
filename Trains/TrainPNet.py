#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：TrainPNet.py
@Author  ：huangxj
@Date    ：2023/9/8 11:27
训练Pnet网络
'''

import os
import sys
from datetime import datetime

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from Loss.loss import ClassLoss, BBoxLoss, LandmarkLoss
from Utiles.Accuracy import accuracy
from Net.PNet import PNet
from Dataset.Utils.mydataset import CustomDataset

# 设置损失值的比例
radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 0.5

# 训练参数值
data_path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\12\\all_data"
batch_size = 384
learning_rate = 1e-3
epoch_num = 30
model_path = '../models'

# 获取P模型
device = torch.device("cpu")
model = PNet()
model.to(device)
# summary(model, (3, 12, 12))

# 获取数据
train_dataset = CustomDataset(data_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 设置优化方法
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)

# 获取学习率衰减函数
scheduler = MultiStepLR(optimizer, milestones=[6, 14, 20], gamma=0.1)

# 获取损失函数
class_loss = ClassLoss()
bbox_loss = BBoxLoss()
landmark_loss = LandmarkLoss()

if __name__ == '__main__':

    # 开始训练
    for epoch in range(epoch_num):
        for batch_id, (img, label, bbox, landmark) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device).long()
            bbox = bbox.to(device)
            landmark = landmark.to(device)
            class_out, bbox_out, landmark_out = model(img)
            cls_loss = class_loss(class_out, label)
            box_loss = bbox_loss(bbox_out, bbox, label)
            landmarks_loss = landmark_loss(landmark_out, landmark, label)
            total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * box_loss + radio_landmark_loss * landmarks_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if batch_id % 100 == 0:
                acc = accuracy(class_out, label)
                print('[%s] Train epoch %d, batch %d, total_loss: %f, cls_loss: %f, box_loss: %f, landmarks_loss: %f, '
                      'accuracy：%f' % (
                          datetime.now(), epoch, batch_id, total_loss, cls_loss, box_loss, landmarks_loss, acc))
        scheduler.step()

        # 保存模型
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.jit.save(torch.jit.script(model), os.path.join(model_path, 'PNet68.pth'))

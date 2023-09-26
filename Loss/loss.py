#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：loss.py
@Author  ：huangxj
@Date    ：2023/9/8 13:47 
'''

import torch.nn as nn
import torch


class ClassLoss(nn.Module):
    """
    是否包含人脸的损失函数
    """

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.keep_ratio = 0.7

    def forward(self, class_out, label):
        # 保留neg 0 和pos 1 的数据，忽略掉part -1, landmark -2
        label = torch.where(label < 0, torch.tensor(-100), label)
        # 求neg 0 和pos 1 的数据70%数据
        valid_label = torch.where(label >= 0, torch.tensor(1), torch.tensor(0))
        num_valid = torch.sum(valid_label)
        keep_num = int((num_valid * self.keep_ratio).cpu().numpy())
        label = torch.squeeze(label)
        # 计算交叉熵损失
        loss = self.entropy_loss(input=class_out, target=label)
        # 取有效数据的70%计算损失
        loss, _ = torch.topk(torch.squeeze(loss), k=keep_num)
        return torch.mean(loss)


class BBoxLoss(nn.Module):
    """
    人脸框偏移的损失函数
    """

    def __init__(self):
        super(BBoxLoss, self).__init__()
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, bbox_out, bbox_target, label):
        # 保留pos 1 和part -1 的数据
        valid_label = torch.where(torch.abs(label) == 1, torch.tensor(1), torch.tensor(0))
        valid_label = torch.squeeze(valid_label)
        # 获取有效值的总数
        keep_num = int(torch.sum(valid_label).cpu().numpy() * self.keep_ratio)
        loss = self.square_loss(input=bbox_out, target=bbox_target)
        loss = torch.sum(loss, dim=1)
        loss = loss * valid_label
        # 取有效数据计算损失
        loss, _ = torch.topk(loss, k=keep_num, dim=0)
        return torch.mean(loss)


class LandmarkLoss(nn.Module):
    """
    关键点的损失函数
    """

    def __init__(self):
        super(LandmarkLoss, self).__init__()
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, landmark_out, landmark_target, label):
        # 只保留landmark数据 -2
        valid_label = torch.where(label == -2, torch.tensor(1), torch.tensor(0))
        valid_label = torch.squeeze(valid_label)
        # 获取有效值的总数
        keep_num = int(torch.sum(valid_label).cpu().numpy() * self.keep_ratio)
        loss = self.square_loss(input=landmark_out, target=landmark_target)
        loss = torch.sum(loss, dim=1)
        loss = loss * valid_label
        # 取有效数据计算损失
        loss, _ = torch.topk(loss, k=keep_num, dim=0)
        return torch.mean(loss)

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：IOU.py
@Author  ：huangxj
@Date    ：2023/9/7 11:55
计算两个方形的IOU值
'''
import numpy as np

def IOU(box, boxes):
    """裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    """
    # box面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes面积,[n,]
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 重叠部分左上右下坐标
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 重叠部分长宽
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # 重叠部分面积
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)
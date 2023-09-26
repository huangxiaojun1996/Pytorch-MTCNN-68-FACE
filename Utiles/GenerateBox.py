#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：GenerateBox.py
@Author  ：huangxj
@Date    ：2023/9/14 15:17
根据神经网络预测结果生成预测框
'''

import numpy as np

def generate_bbox(cls_map, reg, scale, threshold):
    """
    图像坐标反算.得到对应原图的box坐标，分类分数，box偏移量
    cls_map,像素点是否包含人脸的概率
    """
    # pnet大致将图像size缩小2倍
    stride = 2

    cellsize = 12

    # 将置信度高的留下,获取大于阈值的位置
    t_index = np.where(cls_map > threshold)

    # 没有人脸
    if t_index[0].size == 0:
        return np.array([])
    # 偏移量
    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # 对应原图的box坐标，分类分数，box偏移量
    # 原图的box坐标，池化步长 * 坐标index，再除以缩小倍数
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape[n,9]
    return boundingbox.T
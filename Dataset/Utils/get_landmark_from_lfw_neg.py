#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：get_landmark_from_lfw_neg.py
@Author  ：huangxj
@Date    ：2023/9/7 13:54 
'''

import os
import numpy as np
from Dataset.Utils.BBOX import BBox


def get_landmark_from_lfw_neg(txt, data_path, with_landmark=True):
    """
    :param txt: label标签文件
    :param data_path: 图片数据保存目录
    :param with_landmark:bool 是否要保留关键点，默认True
    :return:
    """
    with open(txt, 'r') as f:
        lines = f.readlines()
    result = []
    for line in lines[:500]:
        line = line.strip()
        components = [n for n in line.split(' ') if not n=='']
        # 获取图像路径
        img_path = os.path.join(data_path, components[0]).replace('\\', '/')

        # 人脸box
        # box = (components[1], components[3], components[2], components[4])
        box = (components[1], components[2], components[3], components[4])
        box = [float(_) for _ in box]
        box = list(map(int, box))

        if not with_landmark:
            result.append((img_path, BBox(box)))
            continue
        # 五个关键点(x,y)
        landmark = np.zeros((68, 2))
        for index in range(68):
            rv = (float(components[5 + 2 * index]), float(components[5 + 2 * index + 1]))
            landmark[index] = rv
        result.append((img_path, BBox(box), landmark))
    return result


if __name__ == '__main__':
    txt = 'D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\300w_label.txt'
    data_path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset"
    res = get_landmark_from_lfw_neg(txt, data_path)

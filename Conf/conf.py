#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：conf.py
@Author  ：huangxj
@Date    ：2023/9/8 9:30 
'''

confs = {
    "data-base": "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset",  # 基础路径
    "face-box-train-label": 'wider_face_train.txt',  # 人脸框标签文件，包含人脸图片地址和人脸框的位置
    "face-landmarks-train-label": "300w_label.txt",  # 人脸关键点标签，包含人脸图片路径和关键点位置
}

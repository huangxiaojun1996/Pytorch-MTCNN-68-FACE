#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：Accuracy.py
@Author  ：huangxj
@Date    ：2023/9/8 14:12
准确率
'''

import numpy as np
import torch
import torch.nn.functional as F


def accuracy(class_out, label):
    # 查找neg 0 和pos 1所在的位置
    class_out = class_out.detach().cpu().numpy()
    label = label.cpu().numpy()
    label = np.squeeze(label)
    zeros = np.zeros(label.shape)
    cond = np.greater_equal(label, zeros)
    picked = np.where(cond)
    valid_label = label[picked]
    valid_class_out = class_out[picked]
    # 求neg 0 和pos 1的准确率
    acc = np.sum(np.argmax(valid_class_out, axis=1) == valid_label, dtype='float')
    acc = acc / valid_label.shape[0]
    return acc


def calculate_mse(predicted_keypoints, true_keypoints):
    """
    计算人脸68个关键点的均方误差（MSE）
    :param:
    predicted_keypoints (torch.Tensor): 模型的预测关键点张量，形状为 (batch_size, 68, 2)。
    true_keypoints (torch.Tensor): 真实关键点张量，形状为 (batch_size, 68, 2)。
    :return:
    float: MSE值。
    """
    # 计算每个关键点的均方误差
    squared_errors = (predicted_keypoints - true_keypoints) ** 2
    # 计算所有关键点的MSE均值
    mse = torch.mean(squared_errors)
    return mse.item()  # 将结果转换为标量并返回


if __name__ == '__main__':
    # 使用训练的算法进行预测
    from predict.pred import infer_image
    from Dataset.ibug_300w.FaceLandmarksDataset import FaceBoxLandmarksDataset

    root = "D:\\PYprogram\Pytorch-MTCNN-master\\dataset\\ibug_300W_large_face_landmark_dataset"
    path = "D:\\PYprogram\Pytorch-MTCNN-master\\dataset\\ibug_300W_large_face_landmark_dataset\\labels_ibug_300W_test.xml"
    fbld = FaceBoxLandmarksDataset(root, path)
    all = []

    for i in range(fbld.__len__()):
        image, box, landmark = fbld[i]
        b, l = infer_image(image)
        if not b is None:
            n = l.shape[0]
            mse = calculate_mse(torch.tensor(l.reshape(n,68, 2)), torch.tensor(landmark))
            all.append(mse)

    print(np.array(all).mean())

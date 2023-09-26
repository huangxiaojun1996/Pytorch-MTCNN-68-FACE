#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：ONetDataGenerator.py
@Author  ：huangxj
@Date    ：2023/9/18 10:28
ONet 网络训练数据生成
'''

import os
import pickle
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

from Dataset.DataFormatConverter import convert_data
from Utiles.nms import py_nms
from Dataset.Utils.CombineDataList import combine_data_list, delete_old_img
from Dataset.Utils.CropLandmarkImage import crop_landmark_image
from Utiles.utile import pad, processed_image, save_hard_example, convert_to_square, calibrate_box
from Utiles.GenerateBox import generate_bbox
from Dataset.Utils.mydataset import read_annotation
from Dataset.Utils.get_landmark_from_lfw_neg import get_landmark_from_lfw_neg

# 模型路径
model_path = '../models'

device = torch.device("cpu")
# 获取P模型
pnet = torch.jit.load(os.path.join(model_path, 'PNet.pth'))
pnet.to(device)
pnet.eval()
softmax_p = torch.nn.Softmax(dim=0)

# 获取R模型
rnet = torch.jit.load(os.path.join(model_path, 'RNet.pth'))
rnet.to(device)
rnet.eval()
softmax_r = torch.nn.Softmax(dim=-1)


# 使用PNet模型预测
def predict_pnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32)
    infer_data = torch.unsqueeze(infer_data, dim=0)
    infer_data = infer_data.to(device)
    # 执行预测
    cls_prob, bbox_pred, _ = pnet(infer_data)
    cls_prob = torch.squeeze(cls_prob)
    cls_prob = softmax_p(cls_prob)
    bbox_pred = torch.squeeze(bbox_pred)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


# 使用RNet模型预测
def predict_rnet(infer_data):
    # 添加待预测的图片
    infer_data = torch.tensor(infer_data, dtype=torch.float32)
    infer_data = infer_data.to(device)
    # 执行预测
    cls_prob, bbox_pred, _ = rnet(infer_data)
    cls_prob = softmax_r(cls_prob)
    return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()


def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5, mode='Union')
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
    all_boxes = all_boxes[keep]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes_c


def detect_rnet(im, dets, thresh):
    """通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
    """
    h, w, c = im.shape
    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
    if int(num_boxes) == 0:
        print('P模型检测结果为空！')
        return None, None
    for i in range(int(num_boxes)):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (24, 24))
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        except:
            continue
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None, None

    keep = py_nms(boxes, 0.6, mode='Union')
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes, boxes_c


# 截取pos,neg,part三种类型图片并resize成24x24大小作为RNet的输入
def crop_48_box_image(data_path, filename, min_face_size, scale_factor, p_thresh, r_thresh):
    # pos，part,neg裁剪图片放置位置
    pos_save_dir = os.path.join(data_path, '96/positive')
    part_save_dir = os.path.join(data_path, '96/part')
    neg_save_dir = os.path.join(data_path, '96/negative')
    # RNet数据地址
    save_dir = os.path.join(data_path, '96/')

    # 创建文件夹
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    # 读取标注数据
    data = read_annotation(data_path, filename)
    all_boxes = []
    landmarks = []
    empty_array = np.array([])

    # 使用PNet模型识别图片
    for image_path in tqdm(data['images']):
        assert os.path.exists(image_path), 'image not exists'
        im = cv2.imread(image_path)
        boxes_c = detect_pnet(im, min_face_size, scale_factor, p_thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

        #  使用RNet模型优化
        boxes, boxes_c = detect_rnet(im, boxes_c, r_thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

        all_boxes.append(boxes_c)

    # 把识别结果存放在文件中
    save_file = os.path.join(save_dir, 'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)

    save_hard_example(data_path, 96)


if __name__ == '__main__':
    data_path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\"
    base_dir = os.path.join(data_path, 'ibug_300W_large_face_landmark_dataset')
    filename = os.path.join(data_path, 'wider_face_train.txt')
    min_face_size = 20
    scale_factor = 0.79
    p_thresh = 0.6
    r_thresh = 0.7
    # 获取人脸的box图片数据
    print('开始生成bbox图像数据')
    crop_48_box_image(data_path, filename, min_face_size, scale_factor, p_thresh, r_thresh)
    # 获取人脸关键点的数据
    print('开始生成landmark图像数据')
    # 获取lfw negbox，关键点
    lfw_neg_path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\ibug_300w.txt"
    data_list = get_landmark_from_lfw_neg(lfw_neg_path,base_dir)
    crop_landmark_image(data_path, data_list, 96, argument=True)
    # 合并数据列表
    print('开始合成数据列表')
    combine_data_list(os.path.join(data_path, '96'))
    # 合并图像数据
    print('开始合成图像文件')
    convert_data(os.path.join(data_path, '96'), os.path.join(data_path, '96', 'all_data'))
    # 删除旧数据
    print('开始删除就得图像文件')
    # delete_old_img(data_path, 96)

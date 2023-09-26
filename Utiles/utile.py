#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：utile.py
@Author  ：huangxj
@Date    ：2023/9/14 15:44
存放一些其他方法
'''

import numpy as np
import cv2
import os
from Dataset.Utils.mydataset import read_annotation
import pickle
from tqdm import tqdm
from Utiles.IOU import IOU


def processed_image(img, scale):
    """
    预处理数据，转化图像尺度并对像素归一
    :param img:
    :param scale:
    :return:
    """
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    # 把图片转换成numpy值
    image = np.array(img_resized).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 归一化
    image = (image - 127.5) / 128
    return image


def convert_to_square(box):
    """将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻每个矩形框的最大边长
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5  # x1
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5  # y1
    square_box[:, 2] = square_box[:, 0] + max_side - 1  # x2
    square_box[:, 3] = square_box[:, 1] + max_side - 1  # y2
    return square_box


def save_hard_example(data_path, save_size):
    """
    根据预测的结果裁剪下一个网络所需要训练的图片的标注数据
    :param data_path: 数据的根目录
    :param save_size: 裁剪图片的大小
    :return:
    """
    # 获取原数据集中的标注数据
    filename = os.path.join(data_path, 'wider_face_train.txt')
    data = read_annotation(data_path, filename)

    # 获取原数据集中的图像路径和标注信息
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']

    # 保存裁剪图片数据文件夹
    pos_save_dir = os.path.join(data_path, '%d/positive' % save_size)
    part_save_dir = os.path.join(data_path, '%d/part' % save_size)
    neg_save_dir = os.path.join(data_path, '%d/negative' % save_size)

    # 创建文件夹
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    # 保存图片数据的列表文件
    neg_file = open(os.path.join(data_path, '%d/negative.txt' % save_size), 'w')
    pos_file = open(os.path.join(data_path, '%d/positive.txt' % save_size), 'w')
    part_file = open(os.path.join(data_path, '%d/part.txt' % save_size), 'w')

    # 读取预测结果
    det_boxes = pickle.load(open(os.path.join(data_path, '%d/detections.pkl' % save_size), 'rb'))

    # 保证预测结果和本地数据数量是一样的
    assert len(det_boxes) == len(im_idx_list), "预测结果和真实数据数量不一致"

    # 图片的命名
    n_idx = 0
    p_idx = 0
    d_idx = 0

    # 开始裁剪下一个网络的训练图片
    pbar = tqdm(total=len(im_idx_list))
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        pbar.update(1)
        # 把原标注数据集以4个数据作为一个box进行变形
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)

        # 如果没有预测到数据就调成本次循环
        if dets.shape[0] == 0:
            continue

        # 读取原图像
        img = cv2.imread(im_idx)

        # 把预测数据转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        neg_num = 0
        for box in dets:
            # 获取预测结果中单张图片中的单个人脸坐标，和人脸的宽高
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # 计算iou值
            Iou = IOU(box, gts)

            # 裁剪并统一大小图片
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size), interpolation=cv2.INTER_LINEAR)

            # 划分种类
            if np.max(Iou) < 0.3 and neg_num < 60:
                # 保存negative图片，同时也避免产生太多的negative图片
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # 指定label为0
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # 或者最大iou值的真实box坐标数据
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # 计算偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    # 保存positive图片，同时也避免产生太多的positive图片
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    # 指定label为1
                    pos_file.write(
                        save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    # 保存part图片，同时也避免产生太多的part图片
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    # 指定label为-1
                    part_file.write(
                        save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    pbar.close()
    neg_file.close()
    part_file.close()
    pos_file.close()


def pad(bboxes, w, h):
    """将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:初始图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ey : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    """
    # 人脸矩形的宽和高
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    # 生成空矩阵，用来保存更新后的数据
    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    # 人脸矩形的宽和高备份 - 1
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    tmp_index = np.where(ex > w - 1)  # 找出x2超出图像边界的矩形框
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]  # 矩形宽 + 图像宽 - x2
    ex[tmp_index] = w - 1  # 将图像w-1 赋值给右下坐标x2

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    #
    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    """校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：
      调整后的box是针对原图的绝对坐标
    """

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


def drawLandmark_multiple(img, bbox, landmark):
    """
    :param img: 初始图像
    :param bbox:
    :param landmark:
    :return:
    """
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img


class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        """
        Args:bbox: [x1,x2,y1,y2]
        """
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_
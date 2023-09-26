#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：CropLandmarkImage.py
@Author  ：huangxj
@Date    ：2023/9/7 15:16 
'''

import cv2
from tqdm import tqdm
import random
from Dataset.Utils.BBOX import BBox, flip, rotate
from Utiles.IOU import IOU
import numpy as np
import os


def crop_landmark_image(data_dir, data_list, size, argument=True):
    """
    裁剪并保存带有人脸关键点的图片
    :param data_dir: 根目录
    :param data_list: 训练数据列表 （path,bbox,landmark)
    :param size:
    :param argument:
    :return:
    """
    npr = np.random
    image_id = 0

    # 数据输出路径
    output = os.path.join(data_dir, str(size))
    if not os.path.exists(output):
        os.makedirs(output)

    # 图片处理后输出路径
    dstdir = os.path.join(output, 'landmark')
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)

    # 记录label的txt
    f = open(os.path.join(output, 'landmark.txt'), 'w')
    idx = 0
    for (imgPath, box, landmarkGt) in tqdm(data_list):
        # 存储人脸图片和关键点
        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)

        img_h, img_w, img_c = img.shape
        # 转换成numpy值
        gt_box = np.array([box.left, box.top, box.right, box.bottom])
        # 裁剪人脸图片
        f_face = img[box.top:box.bottom + 1, box.left:box.right + 1]
        try:
            # resize成网络输入大小
            f_face = cv2.resize(f_face, (size, size))
        except Exception as e:
            print(e)
            print('resize成网络输入大小，跳过')
            continue

        # 创建一个空的关键点变量
        landmark = np.zeros((68, 2))
        for index, one in enumerate(landmarkGt):
            # 关键点相对于左上坐标偏移量并归一化，这个就保证了关键点都处于box内
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv

        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(136))

        # 做数据增强处理
        if argument:
            landmark = np.zeros((68, 2))
            # 对图像变换
            idx = idx + 1
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            # 除去过小图像
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(10):
                # 随机裁剪图像大小
                box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                # 随机左上坐标偏移量
                try:
                    delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                    delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                except Exception as e:
                    print(e)
                    print('随机裁剪图像大小，跳过')
                    continue
                # 计算左上坐标
                nx1 = int(max(x1 + gt_w / 2 - box_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - box_size / 2 + delta_y, 0))
                nx2 = nx1 + box_size
                ny2 = ny1 + box_size
                # 除去超过边界的
                if nx2 > img_w or ny2 > img_h:
                    continue
                # 裁剪边框，图片
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                # 计算iou值
                iou = IOU(crop_box, np.expand_dims(gt_box, 0))

                # 只保留pos图像
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    # 关键点相对偏移
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / box_size, (one[1] - ny1) / box_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(136))
                    landmark = np.zeros((68, 2))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    box = BBox([nx1, ny1, nx2, ny2])
                    # 镜像
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(136))
                    # 逆时针翻转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), 5)
                        # 关键点偏移
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(136))

                        # 左右翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(136))
                    # 顺时针翻转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), -5)
                        # 关键点偏移
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(136))

                        # 左右翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(136))
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

        # 开始保存裁剪的图片和标注信息
        for i in range(len(F_imgs)):
            # 剔除数据偏移量在[0,1]之间
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            # 保存裁剪带有关键点的图片
            cv2.imwrite(os.path.join(dstdir, '%d.jpg' % (image_id)), F_imgs[i])
            # 把图片路径和label，还有关键点保存到数据列表上
            landmarks = list(map(str, list(F_landmarks[i])))
            f.write(os.path.join(dstdir, '%d.jpg' % (image_id)) + ' -2 ' + ' '.join(landmarks) + '\n')
            image_id += 1
    f.close()


if __name__ == '__main__':

    from Dataset.Utils.get_landmark_from_lfw_neg import get_landmark_from_lfw_neg
    txt = 'D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\300w_label.txt'
    data_path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset"
    res = get_landmark_from_lfw_neg(txt, data_path)
    crop_landmark_image(data_dir=data_path, data_list=res, size=12)

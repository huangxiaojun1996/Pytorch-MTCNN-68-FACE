#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：landmarkspred.py
@Author  ：huangxj
@Date    ：2023/9/25 13:47
使用mtcnn模型先进行人脸框预测
再使用mobilefacenet进行68点关键点预测
'''

from predict.pred import infer_image
import torch
import cv2
import numpy as np
from Utiles.utile import BBox, drawLandmark_multiple
from Net.mobilefacenet import MobileFaceNet
import time

map_location = 'cpu'

class Faces(object):
    def __init__(self):
        self.mean = np.asarray([0.485, 0.456, 0.406])
        self.std = np.asarray([0.229, 0.224, 0.225])
        self.crop_size = 112
        self.scale = self.crop_size / 112.

        # 载入人脸关键点检测模型
        print('载入MobileFaceNet模型！')
        self.model = MobileFaceNet([112, 112], 136)
        checkpoint = torch.load('../models/mobilefacenet_model_best.pth.tar', map_location=map_location)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def draw(self, image, bbox, landmarks):
        """

        :param image: 原始图像
        :param bbox: 人脸检测框
        :param landmarks: 关键点
        :return:
        """
        image_copy = image.copy()
        for i in range(len(bbox)):
            image_copy = drawLandmark_multiple(image_copy, bbox[i], landmarks[i])
        return image_copy

    def get_face_boxes(self, image):
        """
        :param image: 初始图像
        :return:
        """
        image_copy = image.copy()
        start = time.time()
        faces = infer_image(image_copy)[0]
        end = time.time()
        print("人脸框检测耗时：{} ms".format((end - start) * 1000))
        return faces

    def get_landmarks_from_boxes(self, image, boxes):
        """
        :param image: 初始图像
        :param boxes: 获取的人脸检测框
        :return:
        """
        image_copy = image.copy()
        start = time.time()
        landmarks = []
        bboxs = []
        if len(boxes) == 0:
            print('NO face is detected!')
        else:
            for k, face in enumerate(boxes):
                # 再次过滤人脸预测框小于0.9的
                if face[4] < 0.9:
                    continue

                # 裁剪人脸
                cropface, new_bbox = self.get_crop_face(image_copy, face)
                if cropface is None:
                    continue

                landmark = self.model(cropface).cpu().data.numpy()
                landmark = landmark.reshape(-1, 2)
                landmark = new_bbox.reprojectLandmark(landmark)
                landmarks.append(landmark)
                bboxs.append(new_bbox)

        end = time.time()
        print("关键点检测耗时：{} ms".format((end - start) * 1000))

        return landmarks, bboxs

    def get_crop_face(self, image, box):
        """
        裁剪人脸框，并返回裁剪后的人脸数据
        :param image:
        :param box:[x1,y1,x2,y2]
        :return:
        """
        height, width, _ = image.shape
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h]) * 1.2)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # 保存人脸位置信息
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = image[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (self.crop_size, self.crop_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            print("人脸截取失败")
            face = None
        else:
            test_face = cropped_face.copy()
            test_face = test_face / 255.0

            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            face = torch.from_numpy(test_face).float()
            face = torch.autograd.Variable(face)

        return face, new_bbox

    def main(self, image):
        """
        :param image:
        :return: 返回处理后的图片，以及预测数据
        """
        # 预计在预测前缀加入一个图片大小的判断，避免传入的图像像素大于1080
        # 人脸检测
        faces = self.get_face_boxes(image)
        landmarks, bboxs = self.get_landmarks_from_boxes(image, faces)
        image_copy = self.draw(image, bboxs, landmarks)
        return image_copy, faces, landmarks


if __name__ == '__main__':
    app = Faces()
    image = cv2.imread("F:\\FaceData\\2.jpg")
    image_copy = app.main(image)
    cv2.imshow("show", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

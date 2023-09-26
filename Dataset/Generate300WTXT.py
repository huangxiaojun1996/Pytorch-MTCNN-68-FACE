#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：Generate300WTXT.py
@Author  ：huangxj
@Date    ：2023/9/8 16:46
生成300w数据集的box和landmarks
'''

import os
from PIL import Image
import cv2
import re
import json


def check_file(path: str) -> list:
    """
    :param path: 输入路径
    :return: 返回该路径下的所有文件路径
    """
    os.chdir(path)
    all_file = os.listdir()
    files = []
    for f in all_file:
        if os.path.isdir(f):
            files.extend(check_file(path + '\\' + f))
            os.chdir(path)
        else:
            files.append(os.path.abspath(os.curdir) + '\\' + f)

    return files


def get_points_from_pts(path):
    assert ".pts" in path, "该方法只支持.pts文件"

    with open(path, 'r') as f:
        lines = f.readlines()
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, "".join([line for line in lines]))
    points = matches[0].split("\n")

    return [point.split(" ") for point in points if not point == ""]


def get_box_from_json(path):
    assert ".json" in path, "该方法只支持.json文件"
    with open(path, 'r') as f:
        lines = f.readlines()
    data = "".join([line for line in lines])
    data = json.loads(data)
    return data["shapes"][0]["points"]


def showpoint(path, point_path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    points = get_points_from_pts(point_path)
    for point in points:
        p = [int(float(point[0])), int(float(point[1]))]
        cv2.circle(image, (p[0], p[1]), 1, (0, 255, 0), 1)

    cv2.imshow("show", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':

        # 生成68点数据txt（路径，人脸框，关键点）
        path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\300W"
        paths = check_file(path)
        files = [p for p in paths if "json" in p]
        ff = open("D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\300w_label.txt",'w')

        for file in files:
            # box
            boxs = get_box_from_json(file)
            # landmark
            landmarks = get_points_from_pts(file.replace("json","pts"))
            # 图片保存位置
            f = file.replace("json","png")
            f = f.replace("D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\","")

            strs =''
            for box in boxs:
                for p in box:
                    strs = strs + " " + str(int(p))

            points = ''
            for landmark in landmarks:
                for p in landmark:
                    points = points + " " + p


            ff.write(f + " "+strs + " " + points)
            ff.write("\n")

        ff.close()
        """

    # %数据标注使用
    name = "outdoor"
    id= 298
    path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\300W\\{}_{}.png".format(name,id)
    point_path = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\300W\\{}_{}.pts".format(name,id)
    showpoint(path, point_path)
    """




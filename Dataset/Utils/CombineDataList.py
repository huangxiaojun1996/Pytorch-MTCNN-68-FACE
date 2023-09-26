#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Pytorch-MTCNN-68-FACE 
@File    ：CombineDataList.py
@Author  ：huangxj
@Date    ：2023/9/7 16:47 
'''

import os
import numpy as np
import shutil


def combine_data_list(data_dir):
    """
    将裁剪后的文件夹合并成一个总的txt文件
    最终生成all_data_list.txt文件
    data_dir：data_dir：已经裁剪后的文件夹 12
    """
    npr = np.random
    with open(os.path.join(data_dir, 'positive.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, 'negative.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, 'part.txt'), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir, 'landmark.txt'), 'r') as f:
        landmark = f.readlines()
    with open(os.path.join(data_dir, 'all_data_list.txt'), 'w') as f:
        base_num = len(pos) // 1000 * 1000 + 1
        s1 = '整理前的数据：neg数量：{} pos数量：{} part数量:{} landmark: {} 基数:{}'.format(len(neg), len(pos), len(part),
                                                                            len(landmark), base_num)
        print(s1)
        # 打乱写入的数据顺序，并这里这里设置比例，设置size参数的比例就能得到数据集比例, 论文比例为：3:1:1:2
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=base_num * 3 > len(neg))
        part_keep = npr.choice(len(part), size=base_num, replace=base_num > len(part))
        pos_keep = npr.choice(len(pos), size=base_num, replace=base_num > len(pos))
        landmark_keep = npr.choice(len(landmark), size=base_num * 2, replace=base_num * 2 > len(landmark))

        s2 = '整理后的数据：neg数量：{} pos数量：{} part数量:{} landmark数量：{}'.format(len(neg_keep), len(pos_keep),
                                                                       len(part_keep), len(landmark_keep))
        print(s2)
        with open(os.path.join(data_dir, 'temp.txt'), 'a', encoding='utf-8') as f_temp:
            f_temp.write('%s\n' % s1)
            f_temp.write('%s\n' % s2)
            f_temp.flush()

        # 开始写入列表数据
        for i in pos_keep:
            f.write(pos[i].replace('\\', '/'))
        for i in neg_keep:
            f.write(neg[i].replace('\\', '/'))
        for i in part_keep:
            f.write(part[i].replace('\\', '/'))
        for i in landmark_keep:
            f.write(landmark[i].replace('\\', '/'))


# 合并图像后删除原来的文件
def delete_old_img(old_image_folder, image_size):
    """
    :param old_image_folder:
    :param image_size:
    :return:
    """
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'positive'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'negative'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'part'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, str(image_size), 'landmark'), ignore_errors=True)

    # 删除原来的数据列表文件
    os.remove(os.path.join(old_image_folder, str(image_size), 'positive.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'negative.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'part.txt'))
    os.remove(os.path.join(old_image_folder, str(image_size), 'landmark.txt'))


if __name__ == '__main__':
    data_dir = "D:\\PYprogram\\Pytorch-MTCNN-master\\dataset\\48"
    combine_data_list(data_dir)

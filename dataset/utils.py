# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 12:20
# @Author  : Mingxing Li
# @FileName: fusion.py
# @Software: PyCharm
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.spatial import distance
import numpy as np
import os
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, average_precision_score, \
    roc_auc_score, roc_curve

from sklearn.metrics import roc_curve


class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


def plt_show(images):
    fig = plt.figure(figsize=(16, 16))
    for i, image in enumerate(images):
        fig.add_subplot(len(images)//3, 3, i + 1)
        plt.title(i+1)
        plt.imshow(image)

    plt.show()

def get_next_frame_name(img_name_t1):
    path, filename = os.path.split(img_name_t1)
    next_name = str(int(filename.split('.')[0]) + 1).zfill(4) + '.png'
    img_name_t2 = os.path.join(path, next_name)
    t1_2_t2 = True
    if not os.path.exists(img_name_t2):
        next_name = str(int(filename.split('.')[0]) - 1).zfill(4) + '.png'
        img_name_t2 = os.path.join(path, next_name)
        t1_2_t2 = False

    return img_name_t2, t1_2_t2

def center_crop(image_t1, image_t2):
    w_1 = image_t1.shape[0]
    w_2 = image_t2.shape[0]
    size = min(w_1, w_2)
    if w_1 > w_2:
        x = w_1//2 - w_2//2
        y = x
        image_t1 = image_t1[y:y + size, x:x + size]
    else:
        x = w_2 // 2 - w_1 // 2
        y = x
        image_t2 = image_t2[y:y + size, x:x + size]

    return image_t1, image_t2

def get_image(img_name_t1):
    image_t1 = cv2.imread(img_name_t1)
    image_t1 = cv2.cvtColor(image_t1, cv2.COLOR_BGR2RGB)
    # print(image_t1.shape)


    img_name_t2, t1_2_t2 = get_next_frame_name(img_name_t1)
    image_t2 = cv2.imread(img_name_t2)
    image_t2 = cv2.cvtColor(image_t2, cv2.COLOR_BGR2RGB)
    # print(image_t2.shape)

    if is_face:
        image_t1, image_t2 = center_crop(image_t1, image_t2)
        image_t1 = cv2.resize(image_t1, (224, 224))
        image_t2 = cv2.resize(image_t2, (224, 224))
        # print(image_t1.shape, image_t2.shape)

    if t1_2_t2:
        flow = image_t2 - image_t1
    else:
        flow = image_t1 - image_t2
    distances = distance.cdist(np.reshape(image_t2, newshape=(1, -1)), np.reshape(image_t1, newshape=(1, -1)), 'cosine')  # euclidean cosine
    print('distance:', distances)

    return [image_t1, image_t2, flow]

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    return eer

def show_metrics(labels, outputs, preds=None, describe='Image'):
    if preds is None:
        preds = np.array(np.array(outputs) > 0.5, dtype=int)
    loss = log_loss(labels, outputs)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    AP = average_precision_score(labels, preds)
    AUC = roc_auc_score(labels, outputs)
    EER = calculate_eer(labels, outputs)
    print(describe, 'level:loss:', loss, 'acc:', acc, 'recall:', recall, 'precision:', precision, 'AP:', AP, 'AUC:', AUC,
          'EER:', EER)


if __name__ == "__main__":
    # a = torch.tensor([[0, 1], [2, 3]]).unsqueeze(0).unsqueeze(0)
    # print(a, a.size())
    # tta = Test_time_agumentation()
    # # a = tta.tensor_rotation(a)
    # a = tta.tensor_flip(a)
    # print(a)
    # a = tta.tensor_inverse_flip(a)
    # print(a)
    is_face = False
    
    video_name = '003_000.mp4'
    img_name_t1 = 'C:/Users/BokingChen/Downloads/face/ff++_frames/' + video_name + '/0000.png'
    images = get_image(img_name_t1)

    img_name_t1 = 'C:/Users/BokingChen/Downloads/face/ff++_frames/' + video_name + '/0002.png'
    images += get_image(img_name_t1)

    img_name_t1 = 'C:/Users/BokingChen/Downloads/face/ff++_frames/' + video_name + '/0004.png'
    images += get_image(img_name_t1)

    plt_show(images)

    # img_name_t1 = 'C:/Users/BokingChen/Downloads/face/ff++_frames/' + video_name + '/0395.png'
    # print(get_next_frame_name(img_name_t1))

    pass


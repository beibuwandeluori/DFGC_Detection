"""
Some codes borrowed from https://github.com/jphdotam/DFDC/blob/master/cnn3d/training/datasets_video.py
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 13.10.2020
"""

import cv2
import math
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import os



def load_landmarks(landmarks_file):
    """

    :param landmarks_file: input landmarks json file name
    :return: all_landmarks: having the shape of 5x2 list. represent left eye,
                            right eye, noise, left lip, right lip
    """
    all_landmarks = OrderedDict()
    with open(landmarks_file, "r", encoding="utf-8") as file:
        line = file.readline()
        while line:
            line = json.loads(line)
            all_landmarks[line["image_name"]] = line["landmarks"][0]
            line = file.readline()
    return all_landmarks



class binary_Rebalanced_Dataloader(object):
    def __init__(self, root_path="", video_names=[], phase='train',
                 num_class=2, transform=None):
        assert phase in ['train', 'valid', 'test']
        self.root_path = root_path
        self.video_names = video_names
        self.phase = phase
        self.num_classes = num_class
        self.transform = transform
        self.videos_by_class = self.load_video_name()  # load all video name
        if phase != 'train':
            self.image_name = self.load_image_name()
        else:
            if len(self.videos_by_class['real']) < len(self.videos_by_class['fake']):
                self.smallest_class = 'real'
                self.largest_class = 'fake'
            else:
                self.smallest_class = 'fake'
                self.largest_class = 'real'
            print('The number of real videos is : %d' % len(self.videos_by_class['real']))
            print('The number of fake videos is : %d' % len(self.videos_by_class['fake']))
            self.num_smallest_class = len(self.videos_by_class[self.smallest_class])
            self.num_largest_class = len(self.videos_by_class[self.largest_class])

    def load_video_name(self):
        videos_by_class = {}
        real_videos = []
        fake_videos = []
        for video_name in tqdm(self.video_names):
            if video_name.find('mp4') == -1:
                continue
            if video_name.find('Celeb-real') != -1:  # video is from Celeb-real
                real_videos.append(video_name)
            elif video_name.find('YouTube-real') != -1:  # video is from YouTube-real
                real_videos.append(video_name)
            else:  # video is fake
                fake_videos.append(video_name)
        videos_by_class['real'] = real_videos
        videos_by_class['fake'] = fake_videos
        return videos_by_class

    def load_image_name(self):
        image_names = []
        for video_name in tqdm(self.videos_by_class['real'] + self.videos_by_class['fake']):
            video_path = os.path.join(self.root_path, video_name)
            for image_name in os.listdir(video_path):
                image_names.append(os.path.join(video_name, image_name))
        return image_names

    def __getitem__(self, index):
        if self.phase == 'train':
            if index % 2 == 0:  # load smallest_class face image
                video_name = self.videos_by_class[self.smallest_class][min(index // 2, self.num_smallest_class)]
                video_path = os.path.join(self.root_path, video_name)
                image_name = random.sample(os.listdir(video_path), 1)[0]
                image_path = os.path.join(video_path, image_name)

                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            else:  # load largest_class face image
                video_index_from = math.ceil((index // 2) / self.num_smallest_class * self.num_largest_class)
                # Small epsilon to make sure whole numbers round down (so math.ceil != math.floor)
                video_index_to = math.floor(
                    ((index // 2) + 1) / self.num_smallest_class * self.num_largest_class - 0.0001)
                video_index_to = max(video_index_from, video_index_to)
                video_index_to = min(video_index_to, self.num_largest_class)
                video_index = random.randint(video_index_from, video_index_to)

                video_name = self.videos_by_class[self.largest_class][video_index]
                video_path = os.path.join(self.root_path, video_name)
                image_name = random.sample(os.listdir(video_path), 1)[0]
                image_path = os.path.join(video_path, image_name)

                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            if video_name.find('real') != -1:
                label = 0  # real label is 0
            else:
                label = 1  # fake label is 1
            image = self.transform(image=image)["image"]

        else:
            image_path = os.path.join(self.root_path, self.image_name[index])
            image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            if self.image_name[index].find('real') != -1:
                label = 0  # real label is 0
            else:
                label = 1  # fake label is 1
            image = self.transform(image=image)["image"]
        return image, label

    def __len__(self):
        if self.phase == 'train':  # load Rebalanced image data
            return self.num_smallest_class * 2
        else:  # load all image
            return len(self.image_name)





if __name__ == '__main__':
    height = 256
    width = 256
    import albumentations as A

    train_transform = A.Compose([ A.Resize(height, width)])

    with open('val.txt', 'r') as f:
        train_videos = f.readlines()
        train_videos = [i.strip() for i in train_videos]

    if not os.path.isdir('try'):
        os.mkdir('try')

    dataset = binary_Rebalanced_Dataloader(root_path='/raid/chenhan/Celeb-DF-v2-face',
                                           video_names=train_videos[0:10], phase='train',
                                           transform=train_transform)
    
    for m in range(len(dataset)):
        image, label = dataset[m]
        print (label)
        # print (image)
        # image = image.numpy().transpose(1, 2, 0)
        print(image.shape)
        cv2.imwrite('try/%d_%d.jpg' % (m, label), image)

        # break

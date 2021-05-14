import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import torch
import random
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, CenterCrop,\
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise,\
    GaussianBlur, Resize, Normalize, RandomRotate90, Cutout, GridDropout, CoarseDropout, MedianBlur
from catalyst.data.sampler import BalanceClassSampler
from collections import OrderedDict
import json

try:
    from blending import blend_fake_real_img
except:
    import sys
    thisDir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir
    sys.path.append(os.path.join(thisDir))
    from .blending import blend_fake_real_img


# ———————————————————————————————#
def create_train_transforms(size=300):
    return Compose([
        # ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        HorizontalFlip(),
        GaussNoise(p=0.1),
        GaussianBlur(p=0.1),
        # RandomRotate90(),
        Resize(height=size, width=size),
        # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        # RandomCrop(height=size, width=size),
        # OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.5),
        # OneOf([CoarseDropout(), GridDropout()], p=0.5),
        # ToGray(p=0.2),
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )


def create_val_transforms(size=300):
    return Compose([
        Resize(height=size, width=size),
        # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        # GaussianBlur(blur_limit=3, p=1),
        # CenterCrop(height=size, width=size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
# ———————————————————————————————#

# 同步对应打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def name_resolve(video_name):
    if video_name.split('/')[0] == 'Celeb-real':
        persion_id, video_id = video_name.split('/')[1].split('_')[0], video_name.split('/')[1].split('_')[1]
    elif video_name.split('/')[0] == 'Celeb-synthesis':
        persion_id, video_id = video_name.split('/')[1].split('_')[0], video_name.split('/')[1].split('_')[2]
    else:
        persion_id, video_id = '', video_name.split('/')[1]
    return persion_id, video_id


def total_euclidean_distance(a,b):
    assert len(a) == 68
    return np.sum(np.linalg.norm(a-b, axis=1))
# ———————————————————————————————#


class DFGCDataset(Dataset):
    def __init__(self, root_path='/pubdata/chenby/dataset/Celeb-DF-v2_chenhan/Celeb-DF-v2-face', data_type='train',
                 is_one_hot=False, input_size=300, use_adv=False, use_real_adv=False, test_adv=False, num_classes=2,
                 use_blending=False, seed=2021):
        self.root_path = root_path
        self.data_type = data_type
        self.is_one_hot = is_one_hot
        self.use_adv = use_adv  # fake adv
        self.use_real_adv = use_real_adv
        self.test_adv = test_adv
        self.use_blending = use_blending
        # if num_classes == 2, 0:real face; 1:fake face
        # if num_classes == 3, 0:real face; 1:clean fake face; 2:fake face with adversarial noise
        self.num_classes = num_classes
        self.transform = create_train_transforms(size=input_size) if data_type == 'train' else create_val_transforms(size=input_size)
        this_dir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir
        if data_type != 'test':
            txt_file = os.path.join(this_dir, 'txt/train-list.txt')
        else:
            txt_file = os.path.join(this_dir, 'txt/test-list.txt')

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            self.video_paths = [name.strip().split()[1] for name in lines]
            self.labels = [int(name.strip().split()[0]) for name in lines]
        print('Total video:', len(self.video_paths))
        if data_type != 'test':  # 80% train list for training dataset while 20% for validation dataset
            self.video_paths, self.labels = shuffle_two_array(self.video_paths, self.labels, seed=seed)
            split_index = int(len(self.video_paths) * 0.8)
            if data_type == 'train':
                self.video_paths, self.labels = self.video_paths[:split_index], self.labels[:split_index]
                if self.use_blending:  # load real videos's landmarks for blending
                    self.real_video_landmarks = self.load_real_video_json()
            else:  # val
                self.video_paths, self.labels = self.video_paths[split_index:], self.labels[split_index:]
            print(data_type, len(self.video_paths))

        if data_type != 'train':  # load dataset for valid of test
            self.image_paths, self.labels = self.load_val_or_test_image_paths()
            print(f'{data_type} images length:', len(self.image_paths))

    def get_image_path(self, video_name, label, adv_p=0.5):
        video_path = os.path.join(self.root_path, video_name)
        if self.use_adv and label and random.random() < adv_p:  # fake face 选择对抗样本
            c = random.choice([1, 2, 3, 4, 5])
            video_path = video_path.replace('Celeb-synthesis', f'Celeb-synthesis_adv{c}')
        if self.use_real_adv and label == 0 and random.random() < adv_p:  # real face 选择对抗样本
            video_path = video_path.replace('Celeb-real', 'Celeb-real_adv').\
                replace('YouTube-real', 'YouTube-real_adv')
        image_name = random.sample(os.listdir(video_path), 1)[0]  # 随机挑选一个
        image_path = os.path.join(video_path, image_name)

        return image_path, image_name

    def load_val_or_test_image_paths(self):
        image_names = []
        image_labels = []
        for i, video_name in enumerate(self.video_paths):
            video_path = os.path.join(self.root_path, video_name)
            if self.test_adv and self.labels[i] == 1:
                for c in [1, 2, 3, 4, 5]:
                    video_path_adv = video_path.replace('Celeb-synthesis', f'Celeb-synthesis_adv{c}')
                    samples = os.listdir(video_path_adv)
                    for image_name in samples:
                        image_names.append(os.path.join(video_path_adv, image_name))
                        image_labels.append(self.labels[i])
            elif self.test_adv and self.labels[i] == 0:
                video_path_adv = video_path.replace('Celeb-real', 'Celeb-real_adv').\
                    replace('YouTube-real', 'YouTube-real_adv')
                samples = os.listdir(video_path_adv)
                for image_name in samples:
                    image_names.append(os.path.join(video_path_adv, image_name))
                    image_labels.append(self.labels[i])
            else:
                random.seed(2021)  # 测试和验证时需要固定seed
                samples = random.sample(os.listdir(video_path), 10)  # 每个视频取10帧
                for image_name in samples:
                    image_names.append(os.path.join(video_path, image_name))
                    image_labels.append(self.labels[i])
        return image_names, image_labels

    def load_landmarks(self, landmarks_file):
        """
        :param landmarks_file: input landmarks json file name
        :return: all_landmarks: having the shape of 64x2 list. represent left eye,
                                right eye, noise, left lip, right lip
        """
        all_landmarks = OrderedDict()
        with open(landmarks_file, "r", encoding="utf-8") as file:
            line = file.readline()
            while line:
                line = json.loads(line)
                all_landmarks[line["image_name"]] = np.array(line["landmarks"])
                line = file.readline()
        return all_landmarks

    def search_similar_face(self, video_name, this_landmark):
        persion_id, video_id = name_resolve(video_name)

        min_dist = 99999999
        all_candidat = OrderedDict()
        for real_video_name, all_landmarks in self.real_video_landmarks.items():
            if random.random() > 0.1 or (persion_id + '_' + video_id == real_video_name.split('/')[1]) \
                    or (video_id == real_video_name.split('/')[1]):
                pass
            else:
                frame_name = random.sample(list(all_landmarks.keys()), 1)[0]
                all_candidat[real_video_name + '/' + frame_name] = all_landmarks[frame_name]

        # loop throungh all candidates frame to get best match
        for frame_path, landmark in all_candidat.items():
            candidate_distance = total_euclidean_distance(landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                searched_frame = frame_path
                searched_landmark = landmark

        return os.path.join(self.root_path, searched_frame), searched_landmark

    def load_real_video_json(self):
        print('Loading real video json ...')
        real_video_landmarks = OrderedDict()
        for i, video_name in enumerate(self.video_paths):
            if self.labels[i] == 0:
                json_file = os.path.join(video_name.split('/')[0], 'dlib_landmarks', video_name.split('/')[1] + '.json')
                json_path = os.path.join(self.root_path, json_file)
                if os.path.isfile(json_path):
                    all_landmarks = self.load_landmarks(json_path)
                    if len(all_landmarks) != 0:
                        real_video_landmarks[video_name] = all_landmarks
        return real_video_landmarks

    def __len__(self):
        if self.data_type == 'train':  # load Rebalanced image data
            return len(self.video_paths)
        else:  # load all image
            return len(self.image_paths)

    def get_labels(self):
        return list(self.labels)

    def __getitem__(self, index):
        is_blending = False
        label = self.labels[index]
        if self.data_type == 'train':
            video_name = self.video_paths[index]
            image_path, image_name = self.get_image_path(video_name=video_name, label=label, adv_p=0.75)  # Select the adversarial sample with a probability of 0.75
            if self.use_blending and label:
                json_file = os.path.join(video_name.split('/')[0], 'dlib_landmarks', video_name.split('/')[1] + '.json')
                json_path = os.path.join(self.root_path, json_file)
                video_path = os.path.join(self.root_path, video_name)
                if random.random() > 0.5 and os.path.isfile(json_path):  # Blending with a probability of 0.5
                    all_landmarks = self.load_landmarks(json_path)
                    if len(all_landmarks) != 0:  # A real face and a fake face are combined into a fake face
                        if image_path.find('_adv') == -1:  # A real face and a clean fake face are combined into a fake face
                            image_name = random.sample(list(all_landmarks.keys()), 1)[0]
                            this_landmark = all_landmarks[image_name]
                            searched_path, searched_landmark = self.search_similar_face(video_name, this_landmark)
                            image_path = os.path.join(video_path, image_name)
                            image = blend_fake_real_img(image_path, this_landmark, searched_path, searched_landmark)
                            is_blending = True
                            # print('is_blending')
                            # print(image_path, searched_path)
                            # cv2.imwrite(f'./example/{image_name}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        elif all_landmarks.get(image_name) is not None:  # A real face and a adversarial noise fake face are combined into a fake face
                            this_landmark = all_landmarks[image_name]
                            searched_path, searched_landmark = self.search_similar_face(video_name, this_landmark)
                            image = blend_fake_real_img(image_path, this_landmark, searched_path, searched_landmark)
                            is_blending = True
                            # print('adv is_blending')
                            # print(image_path, searched_path)
                            # cv2.imwrite(f'./example/{image_name}', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if self.num_classes == 3 and label and image_path.find('_adv') != -1:
                label = 2  # 0:real face; 1:clean fake face; 2:fake face with adversarial noise
        else:  # test or val
            image_path = self.image_paths[index]
            if self.num_classes == 3 and label and image_path.find('_adv') != -1:
                label = 2  # 0:real face; 1:clean fake face; 2:fake face with adversarial noise

        if not is_blending:
            image = cv2.imread(image_path)
            # Revert from BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = self.transform(image=image)
        image = data["image"]

        if self.is_one_hot:
            label = one_hot(self.num_classes, label)

        return image, label


if __name__ == '__main__':
    start = time.time()
    xdl = DFGCDataset(data_type='train', is_one_hot=False, input_size=300, use_adv=True, use_real_adv=True,
                      test_adv=False, use_blending=True, num_classes=3)
    print('length:', len(xdl))
    train_loader = DataLoader(xdl, batch_size=1, shuffle=False, num_workers=1)
    # train_loader = DataLoader(xdl, batch_size=8, shuffle=False, num_workers=1,
    #                           sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="upsampling"))
    for i, (img, label) in enumerate(train_loader):
        print(i, img.size(), label.size(), label)
        if i == 1:
            break
    print((time.time()-start))
    pass

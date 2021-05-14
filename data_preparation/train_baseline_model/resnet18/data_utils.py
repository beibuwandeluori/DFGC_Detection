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
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset

import os


def extract_frames(videos_path, detector=None, frame_subsample_count=30, scale=1.3):
    assert detector is not None, 'model is None'

    reader = cv2.VideoCapture(videos_path)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 32
    rgb_frames = OrderedDict()
    pil_frames = OrderedDict()
    for i in range(frames_num):
        for _ in range(frame_subsample_count):
            reader.grab()

        success, frame = reader.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BRG2RGB
        rgb_frames[i] = frame
        frame = Image.fromarray(frame)  # To numpy array
        frame = frame.resize(size=[s // 2 for s in frame.size])
        pil_frames[i] = frame

    rgb_frames = list(rgb_frames.values())
    pil_frames = list(pil_frames.values())
    reader.release()
    crops = []
    for i in range(0, len(pil_frames), batch_size):
        batch_boxes, batch_probs, batch_points = detector.detect(pil_frames[i:i + batch_size], landmarks=True)
        None_array = np.array([], dtype=np.int16)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                pass
            else:
                batch_boxes[index] = None_array
        batch_boxes, batch_probs, batch_points = detector.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                       pil_frames[i:i + batch_size],
                                                                       method="probability")
        # print(batch_probs.shape)
        # print(batch_points.shape)
        # batch_boxes = np.squeeze(batch_boxes, 1)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]  # resize the box
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(rgb_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(rgb_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                crop = rgb_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]
                crops.append(crop)
            else:
                pass
    return crops


class video_Dataloader(Dataset):
    def __init__(self, videos_path, batch_size=32, transform=None, num_class=2, scale=1.3, 
                 frame_subsample_count=30, detector=None):
        assert detector is not None, 'model is None'
        self.videos_path = videos_path

        # extract face images, all face images are rgb images
        self.face_images = extract_frames(self.videos_path, detector=detector, scale=scale,
                                          frame_subsample_count=frame_subsample_count)

        self.batch_size = batch_size
        self.num_class = num_class
        self.transform = transform

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)


class images_Dataloader(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.face_images = self.load_image()
        self.transform = transform

    def load_image(self):
        rgb_frames = []
        for i, filename in enumerate(glob.glob(self.video_path + '/*')):
            frame = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame)
        return rgb_frames

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)


class faces_Dataloader(Dataset):
    def __init__(self, face_images, transform=None):
        self.face_images = face_images
        self.transform = transform

    def __getitem__(self, index):
        image = self.face_images[index]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.face_images)


if __name__ == '__main__':
    height = 256
    width = 256
    from transforms import build_transforms

    transform_train, transform_test = build_transforms(
        height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    with open('try.txt', 'r') as f:
        train_videos = f.readlines()
        train_videos = [i.strip() for i in train_videos]

    # dataset = binary_Rebalanced_Dataloader(root_path='/raid/chenhan/Celeb-DF-v2-face',
    #                                        video_names=train_videos[0:10],
    #                                        resize=(height, width),
    #                                        transform=transform_train)
    # if not os.path.isdir('try'):
    #     os.mkdir('try')
    #
    # for m in range(len(dataset)):
    #     image, label = dataset[m]
    #     print(label)
    #     print(image.shape)
    #     # image = image.numpy().transpose(1, 2, 0)
    #     # print(image.shape)
    #     cv2.imwrite('try/%d_%d.jpg' % (m, label), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # dataset = binary_Rebalanced_Dataloader(root_path='/raid/chenhan/Celeb-DF-v2-face',
    #                                        video_names=train_videos[0:10],
    #                                        resize=(height, width),
    #                                        transform=transform_test)
    #
    # for m in range(len(dataset)):
    #     image, label = dataset[m]
    #     print (label)
    #     print (image)
    #     # image = image.numpy().transpose(1, 2, 0)
    #     # print(image.shape)
    #     # cv2.imwrite('try/%d_%d.jpg' % (m, label), image)

    dataset = Triplet_Rebalanced_Dataloader(root_path='/raid/chenhan/Celeb-DF-v2-face',
                                            video_names=train_videos,
                                            resize=(height, width),
                                            transform=transform_train,
                                            phase='test',
                                            seed=1)
    if not os.path.isdir('try2'):
        os.mkdir('try2')

    print(len(dataset))
    for i, m in enumerate(range(len(dataset))):
        a, p, n = dataset[m]
        print(a.shape)
        cv2.imwrite('try2/%d_a.jpg' % i, cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        cv2.imwrite('try2/%d_p.jpg' % i, cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
        cv2.imwrite('try2/%d_n.jpg' % i, cv2.cvtColor(n, cv2.COLOR_RGB2BGR))

        # break

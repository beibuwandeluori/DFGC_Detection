import os


from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from facenet_pytorch import MTCNN
import dlib
import numpy as np

# import torch
import json
import cv2

import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../dlib_model/shape_predictor_68_face_landmarks.dat')


def save_landmarks(input_dir, save_dir):
    # all_image_names = []

    all_landmarks = OrderedDict()
    all_images = OrderedDict()

    image_names = os.listdir(input_dir)
    for name in image_names:
        if name.find('png') == -1:
            continue
        image = cv2.cvtColor(cv2.imread(os.path.join(input_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)  # BGR2RGB
        image = cv2.resize(image, (256, 256))
        all_images[name] = image

    all_image_names = list(all_images.keys())
    all_images = list(all_images.values())
    for i in range(0, len(all_images)):
        try:
            rect = detector(all_images[i])[0]
            sp = predictor(all_images[i], rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            all_landmarks[all_image_names[i]] = landmarks
        except Exception as e:
            print(e)
            pass

    json_name = input_dir.split('/')[-1] + '.json'
    with open(os.path.join(save_dir, json_name), "w", encoding='utf-8') as out_file:
        for index, name in enumerate(all_landmarks.keys()):
            dict = {"image_name": name, "landmarks": all_landmarks[name].astype(np.int16).tolist()}
            dict = json.dumps(dict) + "\n"
            out_file.write(dict)



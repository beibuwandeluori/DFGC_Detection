"""
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 13.10.2020
"""
from tqdm import tqdm
from generate_landmarks_dlib import save_landmarks
import argparse

from functools import partial
from multiprocessing.pool import Pool

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='Extract face from videos')
           
    parser.add_argument('--video_root_path', type=str, default='../data_structure/Celeb-DF-v2')
    parser.add_argument('--image_root_path', type=str, default='../data_structure/Celeb-DF-v2-face')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()   
    image_root_path = args.image_root_path
    sub_folders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
    
    for sub_folder in sub_folders:
        input_dir = []
        image_path = os.path.join(image_root_path, sub_folder)
        for index, video in tqdm(enumerate(os.listdir(image_path))):
            input_dir.append(os.path.join(image_path, video))
        save_dir = os.path.join(image_path, 'dlib_landmarks')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        print(len(input_dir))
        print('cpu_count: %d' % os.cpu_count())
        with Pool(processes=int(os.cpu_count()/3)) as p:
            with tqdm(total=len(input_dir)) as pbar:
                func = partial(save_landmarks, save_dir=save_dir)
                for v in p.imap_unordered(func, input_dir):
                    pbar.update()


if __name__ == "__main__":
    main()





















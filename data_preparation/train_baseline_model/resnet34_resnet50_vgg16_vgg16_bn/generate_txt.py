"""
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 12.10.2020
"""
import random
from tqdm import tqdm
import argparse

import os

def parse_args():
    parser = argparse.ArgumentParser(description='Genetate train and val txt from videos')
           
    parser.add_argument('--video_root_path', type=str, default='/pubdata/chenby/dataset/Celeb-DF-v2/video')           

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    video_root_path = args.video_root_path
    txt_name = 'List_of_testing_videos.txt'
    sub_folders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']

    with open(os.path.join(video_root_path, txt_name), 'r') as f:
        test_videos = f.readlines()
        test_videos = [i.strip().split(' ')[1] for i in test_videos]

    train_videos = []
    for sub_folder in sub_folders:
        sub_train_videos = []
        video_path = os.path.join(video_root_path, sub_folder)
        for index, video in tqdm(enumerate(os.listdir(video_path))):
            video_name = os.path.join(sub_folder, video)
            if video_name not in test_videos and video_name.find('mp4') != -1:
                sub_train_videos.append(video_name)
        train_videos.append(sub_train_videos)


    train_txt = open('train.txt', 'w')
    val_txt = open('val.txt', 'w')
    for i in range(len(sub_folders)):
        for j in range(len(train_videos[i])):
            if random.random() > 0.2:
                train_txt.write(train_videos[i][j] + '\n')
            else:
                val_txt.write(train_videos[i][j] + '\n')            



if __name__ == "__main__":
    main()

# Helper function for extracting features from pre-trained models
import os
import argparse
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from get_dlib_mask import get_face_mask

from attacker import AttackerTPGD

from models.xception import Xception
from models.resnet34 import ResNet34
from models.vgg16_bn import VGG16_BN
from models.vgg19 import VGG19



class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

        return x

class Resize(nn.Module):

    def __init__(self, input_size=[224, 224]):
        super(Resize, self).__init__()
        self.input_size = input_size

    def forward(self, input):
        x = F.interpolate(input, size=self.input_size, mode='bilinear', align_corners=True)

        return x

class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]

def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class Dataset_DFGC(Dataset):
    def __init__(self, fake_path, is_one_hot=False, transforms=None,frame_num=5,use_mask=True):

        self.is_one_hot = is_one_hot
        self.transforms = transforms
        self.use_mask = use_mask
        self.img_paths = []
        self.labels = []

        sub_dirs = os.listdir(fake_path)

        for sub_dir in sub_dirs:
            if sub_dir.find('.mp4') != -1:
                sub_path=os.path.join(fake_path,sub_dir)
                fake_paths=os.listdir(sub_path)
                rand_idx = np.random.randint(0,len(fake_paths),size=min(frame_num,len(fake_paths)))
                for idx in rand_idx:
                    self.img_paths.append(os.path.join(sub_path,fake_paths[idx]))
                    self.labels.append(1)


    def __len__(self):
        l = len(self.labels)
        return l

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        image = cv2.imread(img_name)
        # Revert from BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.use_mask :
            mask = get_face_mask(image).astype(np.uint8)
            mask = np.transpose(mask, axes=[2, 0, 1])

        if self.transforms is not None:
            image = self.transforms(Image.fromarray(image))
        else:
            image = np.transpose(image.astype(np.float32), axes=[2, 0, 1])
            image = image / 255.0

        label = self.labels[index]
        if self.is_one_hot:
            label = one_hot(2, label)
        if self.use_mask:
            return image, mask, label, img_name
        return image, label, img_name

class Ensemble(nn.Module):
    def __init__(self, model1, model2=None, model3=None,model4=None):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, x):
        logits1 = self.model1(x)
        logits2 = self.model2(x)[:, 1:]  # 为假的概率
        logits3 = self.model3(x)[:, 1:]  # 为假的概率
        logits4 = self.model4(x)[:, 1:]  # 为假的概率

        # fuse logits
        # logits_e = (logits1 + logits2) / 2
        logits_e = (logits1 + logits2 + logits3 + logits4) / 4

        return logits_e

def parse_args():
    parser = argparse.ArgumentParser(description='generate adversial faces')
    parser.add_argument('--input_path', type=str, default='/pubdata/chenby/dataset/Celeb-DF-v2_chenhan/Celeb-DF-v2-face/Celeb-synthesis/')
    parser.add_argument('--use_mask', default=False, action='store_true',)
    parser.add_argument('--frames', type=int, default=2)
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--steps', default=20, type=int, help='iteration steps')
    parser.add_argument('--max_norm', default=8, type=float, help='Linf limit')
    parser.add_argument('--div_prob', default=0.5, type=float, help='probability of diversity')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    # set dataset

    fake_path = args.input_path
    output_dir = args.input_path + '_adv2'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dataset = Dataset_DFGC(fake_path, is_one_hot = False, transforms = None, frame_num = args.frames, use_mask = args.use_mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    #model load
    model = Xception()
    model.fc = nn.Linear(2048, 1)
    model_path = './weights/weights.ckpt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = nn.Sequential(
        Resize(input_size=[299, 299]),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        model
    )
    model2 = ResNet34(2, droprate=0)
    model_path2 = './weights/ResNet34_png.pth'
    model2.load_state_dict(torch.load(model_path2, map_location='cpu'))
    model2 = nn.Sequential(
        Resize(input_size=[256, 256]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model2
    )
    model3 = VGG16_BN(2, droprate=0)
    model_path3 = './weights/VGG16_BN.pth'
    model3.load_state_dict(torch.load(model_path3, map_location='cpu'))
    model3 = nn.Sequential(
        Resize(input_size=[224, 224]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model3
    )
    model4 = VGG19(2, droprate=0)
    model_path4 = './weights/VGG19.pth'
    model4.load_state_dict(torch.load(model_path4, map_location='cpu'))
    model4 = nn.Sequential(
        Resize(input_size=[224, 224]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model4
    )
    
    model = Ensemble(model, model2, model3, model4)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # set attacker
    attackers=AttackerTPGD( eps=args.max_norm/255.0, alpha=2/255.0, steps=args.steps, low=1.0, high=1.2,
                           div_prob=args.div_prob,device=torch.device('cuda'))

    #attack
    for ind, (img,mask, label_true, filenames) in enumerate(loader):
        # run attack
        adv = attackers.attack(model,img.cuda(), label_true.cuda(),mask.cuda())
        # save results
        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()
            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0
            first,last = os.path.split(filename)
            first = os.path.split(first)[-1].split('.')[0] + '.mp4'
            if not os.path.isdir(os.path.join(output_dir, first)):
                os.makedirs(os.path.join(output_dir, first))
            file_name = first+'/'+last

            print(ind, 'Attack on {}:'.format(file_name))
            print(ind, 'Max: {0:.0f}, Mean: {1:.2f}'.format(np.max(delta_img), np.mean(delta_img)))

            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]


            out_filename = os.path.join(output_dir, file_name)

            cv2.imwrite(out_filename, out_img)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    main()
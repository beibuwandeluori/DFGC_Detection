import os
import os.path
import torch
import cv2
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import torchvision.transforms as Transforms
from torch.utils.data import dataset, dataloader
import torch.nn as nn
from .efficientnet import TransferModel
import json
import torch.nn.functional as F


class FolderDataset(dataset.Dataset):
    def __init__(self, img_folder, face_info, input_size=300):
        self.img_folder = img_folder
        self.imgNames = sorted(os.listdir(img_folder))
        # REMEMBER to use sorted() to ensure correct match between imgNames and predictions
        # do NOT change the above two lines

        self.face_info = face_info
        self.transform = Transforms.Compose([
                            Transforms.Resize((input_size, input_size)),
                            Transforms.ToTensor(),
                            Transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.imgNames)

    def read_crop_face(self, img_name, img_folder, info):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        img_name = os.path.splitext(img_name)[0]  # exclude image file extension (e.g. .png)
        # landms = info[img_name]['landms']
        box = info[img_name]['box']
        height, width = img.shape[:2]
        # enlarge the bbox by 1.3 and crop
        scale = 1.3
        # if len(box) == 2:
        #     box = box[0]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(int(center_x - size_bb // 2), 0) # Check for out of bounds, x-y top left corner
        y1 = max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)

        cropped_face = img[y1:y1 + size_bb, x1:x1 + size_bb]
        return cropped_face

    def __getitem__(self, idx):
        img_name = self.imgNames[idx]
        # Read-in images are full frames, maybe you need a face cropping.
        img = self.read_crop_face(img_name, self.img_folder, self.face_info)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class Resize(nn.Module):

    def __init__(self, input_size=[224, 224]):
        super(Resize, self).__init__()
        self.input_size = input_size

    def forward(self, input):
        x = F.interpolate(input, size=self.input_size, mode='bilinear', align_corners=True)

        return x


def load_models(model_names, model_paths, device_id=0):
    thisDir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir
    models = []
    for i in range(len(model_names)):
        model = TransferModel(model_names[i], num_out_classes=3)
        model_path = model_paths[i]
        model.load_state_dict(torch.load(os.path.join(thisDir, model_path), map_location='cpu'))
        print('Load model ', i, 'in:', model_path)
        model.eval()
        model.cuda(device_id)
        models.append(model)

    return models


class Model():
    def __init__(self, device_id=0):
        thisDir = os.path.dirname(os.path.abspath(__file__))  # use this line to find this file's dir

        self.is_ensemble = False
        self.tta = True
        self.input_size = 300
        self.device_id = device_id
        if not self.is_ensemble:
            # init and load your model here
            model = TransferModel('efficientnet-b3', num_out_classes=3)
            model_path = os.path.join(thisDir, 'efn-b3_3c_60_acc0.9975.pth')  # you can replace it by your weight
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f'Load model in {model_path}')
            model.eval()
            model.cuda(self.device_id)
            self.model = model
        else:
            self.models = load_models(model_names=['efficientnet-b3', 'efficientnet-b2'],
                                      model_paths=['60_acc0.9975.pth',
                                                   '80_acc0.9964.pth'], device_id=device_id)

        # determine your own batchsize based on your model size. The GPU memory is 16GB
        # relatively larger batchsize leads to faster execution.
        self.batchsize = 20

    def run(self, input_dir, json_file):
        with open(json_file, 'r') as load_f:
            json_info = json.load(load_f)
        dataset_eval = FolderDataset(input_dir, json_info, input_size=self.input_size)
        dataloader_eval = dataloader.DataLoader(dataset_eval, batch_size=self.batchsize,
                                                shuffle=False, num_workers=4)
        # USE shuffle=False in the above dataloader to ensure correct match between imgNames and predictions
        # Do set drop_last=False (default) in the above dataloader to ensure all images processed

        print('Detection model inferring ...')
        prediction = []
        with torch.no_grad():  # Do USE torch.no_grad()
            for imgs in tqdm(dataloader_eval):
                imgs = imgs.to(f'cuda:{self.device_id}')
                # outputs = self.model(imgs)
                # print(imgs.shape)
                # outputs = nn.Softmax(dim=1)(self.model(imgs))[:, 1]
                # TTA
                if not self.is_ensemble:
                    # TTA
                    if self.tta:
                        outputs = (nn.Softmax(dim=1)(self.model(imgs)) +
                                   nn.Softmax(dim=1)(self.model(imgs.flip(dims=(2,)))) +
                                   nn.Softmax(dim=1)(self.model(imgs.flip(dims=(3,)))))/3.0
                    else:
                        outputs = nn.Softmax(dim=1)(self.model(imgs))
                    prediction.append(1-outputs[:, 0])
                    # print(prediction)
                else:
                    for i, model in enumerate(self.models):
                        if i == 0:
                            if self.tta:
                                outputs = (nn.Softmax(dim=1)(model(imgs)) +
                                           nn.Softmax(dim=1)(model(imgs.flip(dims=(2,)))) +
                                           nn.Softmax(dim=1)(model(imgs.flip(dims=(3,))))) / 3.0
                            else:
                                outputs = nn.Softmax(dim=1)(model(imgs))
                        else:
                            if self.tta:
                                outputs += (nn.Softmax(dim=1)(model(imgs)) +
                                           nn.Softmax(dim=1)(model(imgs.flip(dims=(2,)))) +
                                           nn.Softmax(dim=1)(model(imgs.flip(dims=(3,))))) / 3.0
                            else:
                                # outputs += nn.Softmax(dim=1)(model(imgs))
                                outputs += nn.Softmax(dim=1)(model(Resize(input_size=[260, 260])(imgs)))
                    outputs /= len(self.models)
                    prediction.append(1-outputs[:, 0])
                    # print(prediction)

        prediction = torch.cat(prediction, dim=0)
        prediction = prediction.cpu().numpy()
        prediction = prediction.squeeze().tolist()
        assert isinstance(prediction, list)
        assert isinstance(dataset_eval.imgNames, list)
        assert len(prediction) == len(dataset_eval.imgNames)

        return dataset_eval.imgNames, prediction

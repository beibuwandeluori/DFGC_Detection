import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.data import BalanceClassSampler
from torch.autograd import Variable
import time

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from utils.utils import Logger, AverageMeter, calculate_metrics, Test_time_agumentation
from network.models import model_selection
from dataset.dataset import DFGCDataset
from loss.losses import LabelSmoothing

# 9 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img))
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img))
    # 2*3=6
    for flip_img in [img, flip_imgs[0]]:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += activation(model_(rot_flip_img))

    outputs /= 9

    return outputs

def eval_model(model, epoch, eval_loader, is_save=True, is_tta=False, metric_name='acc'):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels = []
    outputs = []
    val_criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            if i > 0:
                eval_process.set_description( "Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                              (epoch, losses.avg.item(), accuracies.avg.item()))
            img, label = Variable(img.cuda(device_id)), Variable(label.cuda(device_id))
            if not is_tta:
                y_pred = model(img)
                y_pred = nn.Softmax(dim=1)(y_pred)
            else:
                y_pred = TTA(model, img, activation=nn.Softmax(dim=1))
            outputs.append(1-y_pred[:, 0])
            labels.append(label)
            loss = val_criterion(y_pred, label)
            acc = calculate_metrics(y_pred.cpu(), label.cpu(), metric_name=metric_name)
            losses.update(loss.cpu(), img.size(0))
            accuracies.update(acc, img.size(0))

    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    labels[labels > 0] = 1
    AUC = roc_auc_score(labels, outputs)
    print('AUC:', AUC)
    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'acc': format(accuracies.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))

    return AUC  # accuracies.avg

def train_model(model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)
    for i, (XI, label) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" % (epoch, losses.avg.item(), accuracies.avg.item()))

        x = Variable(XI.cuda(device_id))
        label = Variable(label.cuda(device_id))
        # label = Variable(torch.LongTensor(label).cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, label)
        acc = calculate_metrics(nn.Softmax(dim=1)(y_pred).cpu(), label.cpu())
        losses.update(loss.cpu(), x.size(0))
        accuracies.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}". format(losses.avg, accuracies.avg))


if __name__ == '__main__':
    root_path = '/pubdata/chenby/dataset/Celeb-DF-v2_chenhan/Celeb-DF-v2-face'
    batch_size = 8
    test_batch_size = 64
    input_size = 300
    epoch_start = 0
    num_epochs = 85
    device_id = 0  # set the gpu id
    lr = 1e-3
    use_adv = True  # use fake adv
    use_real_adv = False  # use real adv
    use_blending = True
    seed = 2021  # 默认是2021
    model_name = 'efficientnet-b3'
    writeFile = 'output/logs/' + model_name + '_3c_LS_Celeb-DF_' + str(input_size)
    store_name = 'output/weights/' + model_name + '_3c_LS_Celeb-DF_' + str(input_size)
    if use_adv:
        writeFile += '_adv'
        store_name += '_adv'
    if use_blending:
        # blending:size random from 256-300，p=0.5 blending p=0.75 adv, upsample
        writeFile += '_blending'
        store_name += '_blending'
    if store_name and not os.path.exists(store_name):
        os.makedirs(store_name)

    model_path = None
    # model_path = './output/weights/efficientnet-b3_3c_LS_Celeb-DF_300_adv_blending/80_acc0.9972.pth'
    # Load model
    model, *_ = model_selection(modelname=model_name, num_out_classes=3)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    model = model.cuda(device_id)
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)
    is_train = True
    if is_train:
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=4e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)

        xdl = DFGCDataset(root_path, data_type='train', is_one_hot=True, input_size=input_size, use_adv=use_adv,
                          use_real_adv=use_real_adv, use_blending=use_blending, seed=seed, num_classes=3)
        # train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=False, num_workers=4,
                                  sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="upsampling"))
        train_dataset_len = len(xdl)

        xdl_eval = DFGCDataset(root_path, data_type='val', is_one_hot=False, input_size=input_size, seed=seed)
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        best_acc = 0.5 if epoch_start == 0 else eval_model(model, epoch_start-1, eval_loader, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            train_model(model, criterion, optimizer, epoch)
            if epoch % 20 == 0 or epoch == num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader)
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc))
            print('current best acc:', best_acc)
    else:
        input_size = 300
        seed = 2021
        batch_size = 64
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = DFGCDataset(root_path, data_type='test', input_size=input_size, test_adv=False, seed=seed, num_classes=3)
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        eval_model(model, epoch_start, test_loader, is_save=False, is_tta=False, metric_name='acc')
        print('Total time:', time.time() - start)








"""
Author: HanChen
Date: 15.10.2020
"""
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

import random
import numpy as np
import pandas as pd

from vgg19 import VGG19
from transforms import build_transforms
from metrics import get_metrics
from celeb_df_v2_dataset import binary_Rebalanced_Dataloader

import os



######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network

def parse_args():
    parser = argparse.ArgumentParser(description='Training network on celeb_df_v2_dataset')
           
    parser.add_argument('--root_path', type=str, default='/pubdata/chenhan/Celeb-DF-v2-face')           
    parser.add_argument('--save_path', type=str, default='./save_result')           
    parser.add_argument('--gpu_id', type=int, default=5)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=1200)
    parser.add_argument('--val_epochs', type=int, default=40)
    parser.add_argument('--adjust_lr_epochs', type=int, default=40)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--val_batch_size', type=int, default=256)

   
    args = parser.parse_args()


    return args



def main():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    with open('train.txt', 'r') as f:
        train_videos = f.readlines()
        train_videos = [i.strip() for i in train_videos]
    with open('val.txt', 'r') as f:
        val_videos = f.readlines()
        val_videos = [i.strip() for i in val_videos]
        

    train_dataset = binary_Rebalanced_Dataloader(root_path=args.root_path, video_names=train_videos, phase='train', 
                                                num_class=args.num_class, transform=transform_train)

    test_dataset = binary_Rebalanced_Dataloader(root_path=args.root_path, video_names=val_videos, phase='test',
                                                num_class=args.num_class, transform=transform_test)

    print('Test Images Number: %d' % len(test_dataset))
    print('All Train videos Number: %d' % (train_dataset.num_smallest_class + train_dataset.num_largest_class))
    print('Use Train videos Number: %d' % len(train_dataset))

    model = VGG19(args.num_class, droprate=args.droprate).cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, num_workers=3, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.val_batch_size,
                                              drop_last=True, num_workers=3, pin_memory=True)

    best_loss = 100.0
    loss_name = ['BCE']
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        running_loss = {loss: 0 for loss in loss_name}
        model.train(True)  # Set model to training mode
        # Iterate over data (including images and labels).
        for images, labels in train_loader:
            # wrap them in Variable
            images = Variable(images.cuda().detach())
            labels = Variable(labels.cuda().detach())

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)

            # update the parameters
            loss.backward()
            optimizer.step()

            running_loss['BCE'] += loss.item()
            # break

        print('Epoch: {:g}, Step: {:g}, BCE: {:g} '.
              format(epoch, len(train_loader), *[running_loss[name] / len(train_loader) for name in loss_name]))


        if (epoch + 1) % args.adjust_lr_epochs == 0:
            scheduler.step()

        if (epoch + 1) % args.val_epochs == 0 and epoch>500:
            model.train(False)
            model.eval()
            running_loss = {loss: 0 for loss in loss_name}
            correct = {class_index: 0 for class_index in range(args.num_class)}
            total = {class_index: 0 for class_index in range(args.num_class)}

            y_true = np.array([])
            y_pred = np.array([])
            for images, labels in test_loader:
                # wrap them in Variable
                images = Variable(images.cuda().detach())
                y_true = np.insert(y_true, 0, labels.numpy())
                labels = Variable(labels.cuda().detach())

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = model(images)
                # Calculate loss
                loss = criterion(outputs, labels)

                prediction = torch.nn.functional.softmax(outputs, dim=-1)
                y_pred = np.insert(y_pred, 0, prediction.cpu().detach().numpy()[:, 1])
                prediction = torch.argmax(prediction, dim=-1)
                res = prediction == labels
                for label_idx in range(len(labels)):
                    label_single = labels[label_idx].item()
                    correct[label_single] += res[label_idx].item()
                    total[label_single] += 1

                
                # statistics loss
                running_loss['BCE'] += loss.item()

            recall, precision, auc, EER, f1, accuracy = get_metrics(y_true, y_pred)

            df_acc = pd.DataFrame()
            df_acc['epoch'] = [epoch]
            df_acc['BCE'] = running_loss['BCE'] / len(test_loader)
            df_acc['recall'] = recall
            df_acc['precision'] = precision
            df_acc['auc'] = auc
            df_acc['EER'] = EER
            df_acc['f1'] = f1
            df_acc['accuracy'] = accuracy

            sum_correct = 0
            sum_total = 0
            for idx in range(args.num_class):
                sum_correct += correct[idx]
                sum_total += total[idx]
                df_acc[args.class_name[idx]] = correct[idx] / total[idx]
            avg_acc = sum_correct/sum_total
            df_acc['Acc'] = avg_acc
            if epoch+1 != args.val_epochs + 500:
                df_acc.to_csv('%s/report/VGG19.csv' % args.save_path, mode='a', index=None, header=None)
            else:
                df_acc.to_csv('%s/report/VGG19.csv' % args.save_path, mode='a', index=None)

            print('Epoch: {:g}, Step: {:g}, ACC: {:g}, BCE: {:g}'.
                  format(epoch, len(test_loader), avg_acc, *[running_loss[name] / len(test_loader) for name in loss_name]))
                  
            if best_loss > running_loss['BCE'] / len(test_loader):
                best_loss = running_loss['BCE'] / len(test_loader)
                save_network(model, '%s/models/VGG19.pth' % args.save_path)



if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path)

    main()

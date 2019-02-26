# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:32:08 2019

Resnet18: image to 64 dim vector

@author: illya
"""

#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2017-08-16

from __future__ import print_function
import argparse
import os
os.chdir("/users/wangkai/Downloads/271project/")
from simple_train import *
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset

from resnet_model import ResNet18

from characters import characters
from imageread import imageread
import torchvision
from skimage.transform import resize


def get_data(path): # generate random samples
#    X sampls, Y labels
#    Description = np.load( "onehot.npy" )
    Characters, petid, Y = characters(path)
#    X = np.hstack((Characters, Description))
    X = Characters
    return X, Y

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    flag = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        output, out_64 = model(data)
        
        if flag == 0:
            features_64 = out_64.detach().cpu().numpy()
            flag = 1
        else:
            out_64 = out_64.detach().cpu().numpy()
            features_64 = np.vstack((features_64,out_64))
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(label), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return features_64

def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        flag = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device=device, dtype=torch.long)
            output, out_64 = model(data)
            
            if flag == 0:
                features_64 = out_64.detach().cpu().numpy()
                flag = 1
            else:
                out_64 = out_64.detach().cpu().numpy()
                features_64 = np.vstack((features_64,out_64))
            
            criterion = nn.CrossEntropyLoss()
            test_loss += test_loader.batch_size * criterion(output, label).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return features_64

    
class PetDataset(Dataset):
    """ img: 128*128, label: 0~5 """
    def __init__(self, image, label):
        self.image = np.zeros((len(image),32,32))
        for index, img in enumerate(image):
            self.image[index,:,:] = resize(img, (32,32), anti_aliasing=True)
#        label_onehot = np.zeros((label.size, label.max()+1))
#        label_onehot[np.arange(label.size), label] = 1
#        self.label = label_onehot
        self.label = label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        x = self.image[index,:,:]
        return (np.expand_dims(x, axis=0), self.label[index])
        
        
        
def img_64(EPOCH = 10, LR=0.1, Momentum=0.5, no_cuda=False, save_model=False):
    train_path = "/train.csv"
#    test_path = "petfinder-adoption-prediction/test.csv"
    train_img_path = "/data_organized_hist"
#    test_img_path = "testdata_organized_hist/"
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    X, petid, Y = characters(train_path) # load data

    train_label = Y[0:10000]
    test_label = Y[10000:14993]
    
    #IMG = imageread(petid, train_img_path)

    IMG =
    train_img = IMG[0:10000,:,:]
    test_img = IMG[10000:14993,:,:]
    
    train_dataset = PetDataset(train_img, train_label)
    test_dataset = PetDataset(test_img, test_label)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 64, 
        shuffle = False, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 100, 
        shuffle = False, **kwargs)
    
    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=Momentum)
    
    for epoch in range(1, EPOCH + 1):
        train_img_64 = train(model, device, train_loader, optimizer, epoch)
        test_img_64 = test(model, device, test_loader)
        
    
    if (save_model):
        torch.save(model.state_dict(),"pet_resnet.pt")
    
#    train_img_64 = train_out_64.detach().cpu().numpy()
#    test_img_64 = test_out_64.detach().cpu().numpy()
    
    return train_img_64, test_img_64


if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                        help='input batch size for training (default: 64)')
#    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
#                        help='input batch size for testing (default: 100)')
#    parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                        help='number of epochs to train (default: 100)')
#    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
#                        help='learning rate (default: 0.1)')
#    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
#                        help='SGD momentum (default: 0.5)')
#    parser.add_argument('--no-cuda', action='store_true', default=False,
#                        help='disables CUDA training')
#    parser.add_argument('--seed', type=int, default=1, metavar='S',
#                        help='random seed (default: 1)')
#    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                        help='how many batches to wait before logging training status')
#    
#    parser.add_argument('--save-model', action='store_true', default=False,
#                        help='For Saving the current Model')
#    
#    args = parser.parse_args()

    img_64(EPOCH = 1, LR=0.1, Momentum=0.5, no_cuda=False, save_model=False)
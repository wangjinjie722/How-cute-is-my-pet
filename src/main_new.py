# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:32:08 2019

3 channel grayscale image: 3,128,128
extract feature by alexnet
classify to 5 classes

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
os.chdir("Z://00_UCSD/ECE 271B/Project/") 

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

#from resnet_model_modified import resnet_18

from characters import characters
from load_petdata import PetDataset
from pretrained import Vgg16_Feature, Alexnet_Feature


#%%
#def im2feature(model,device,dataloader):
#    model.eval()
#    with torch.no_grad():
##        flag = 0
#        results = []
#        for data, label in dataloader:
#            data, label = data.to(device), label.to(device=device, dtype=torch.long)
#            features = model(data)
#            results.append(features)
#    return results

#%%


def train(model, device, train_loader):
    model.eval()
    with torch.no_grad():
        flag = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device=device, dtype=torch.long)
            out = model(data)
            # shape need to change with model
            out_vec = out.detach().cpu().numpy().reshape(len(label),256*3*3) 
            if flag == 0:
#                pca = PCA(n_components = 256, svd_solver='full')
#                pca.fit(out_vec)
#                results = pca.transform(out_vec)
                results = out_vec
                flag = 1
            else:
#                features = pca.transform(out_vec)
#                results = np.vstack((results,features))
                results = np.vstack((results,out_vec))
            
            if batch_idx % 10 == 0:
                print('Train: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(label), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))
            
    return results#, pca
            
#%%
def test(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        flag = 0
        results = []
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device=device, dtype=torch.long)
            out = model(data)
            # shape need to change with model
            out_vec = out.detach().cpu().numpy().reshape(len(label),256*3*3) 
            if flag == 0:
#                results = pca.transform(out_vec)
                results = out_vec
                flag = 1
            else:
#                features = pca.transform(out_vec)
#                results = np.vstack((results,features))
                results = np.vstack((results,out_vec))
            
            if batch_idx % 10 == 0:
                print('Test: [{}/{} ({:.0f}%)]'.format(
                    batch_idx * len(label), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader)))
    
    return results

#%%
def classifier(train_img_feature, test_img_feature, train_dataset, test_dataset):
#    clf = svm.SVC(gamma='scale') # 0.71
    clf = AdaBoostClassifier(n_estimators = 50) # 0.62
#    clf = GradientBoostingClassifier(n_estimators = 20) # 0.62
    clf.fit(train_img_feature, train_dataset.label)
    
    Train_Y_predict = clf.predict(train_img_feature)
    Train_error = np.sum(train_dataset.label != Train_Y_predict) / len(train_dataset.label)
    print('Train error:', Train_error)
    
    Test_Y_predict = clf.predict(test_img_feature)
    Test_error = np.sum(test_dataset.label != Test_Y_predict) / len(test_dataset.label)
    print('Test error:', Test_error)
    
    return Train_error, Test_error

#%%
    
def main(no_cuda=False, save_model=False):
    train_csv_path = "petfinder-adoption-prediction/train.csv"
    train_img_path = "data_organized_hist/"
#    test_path = "petfinder-adoption-prediction/test.csv"
#    test_img_path = "testdata_organized_hist/"
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    train_dataset = PetDataset(train_img_path, train_csv_path, istrain = 1)
    test_dataset = PetDataset(train_img_path, train_csv_path, istrain = 0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 100, 
        shuffle = False, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 100, 
        shuffle = False, **kwargs)
    
    model = Alexnet_Feature().to(device).double()
        
#    train_img_feature, pca = train(model, device, train_loader)
    train_img_feature = train(model, device, train_loader)
    test_img_feature = test(model, device, test_loader)
    
    Train_error, Test_error = classifier(
            train_img_feature, test_img_feature, train_dataset, test_dataset)
    
    if (save_model):
        torch.save(model.state_dict(),"pet_vgg.pt")
    
#    train_img_64 = train_out_64.detach().cpu().numpy()
#    test_img_64 = test_out_64.detach().cpu().numpy()
    
    return train_img_feature, test_img_feature

#%%
if __name__ == "__main__":
    main(no_cuda=False, save_model=False)
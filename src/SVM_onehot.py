# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:32:08 2019

SVM - one-hot description

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
import csv
import os
os.chdir("/users/wangkai/Downloads/271project/")

import matplotlib.pyplot as plt
import numpy as np

#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

#from nnmodel import Net, train, test

from characters import characters
from NN_image import img_64

def get_data(path): 
############  X sampls, Y labels ###########
    train_img_64, test_img_64 = img_64(EPOCH = 5, LR=0.1, Momentum=0.9, no_cuda=False, save_model=False)
    Characters, petid, Y = characters(path)
    Description = np.load( "train_hot.npy" )
    Img_64 = np.vstack((train_img_64, test_img_64))
    X = np.hstack((Img_64, Characters, Description))
    return X, Y



def main(args):
    train_path = "/users/wangkai/Downloads/271project/train.csv"
#    test_path = "petfinder-adoption-prediction/test.csv"
    
#    SVM
    X, Y = get_data(train_path) # training data
    Train_X = X[0:10000,:]
    Train_Y = Y[0:10000]
    Test_X = X[10000:14993,:]
    Test_Y = Y[10000:14993]
    
#    model = svm.SVC(gamma='scale') # 0.71
#    model = svm.LinearSVC() #0.69
#    model = AdaBoostClassifier(n_estimators = 30) # 0.61
    model = GradientBoostingClassifier(n_estimators=50) # 0.61 
#    model = KNeighborsClassifier(n_neighbors=5) # 0.68
    
#    clf1 = AdaBoostClassifier(n_estimators=500) # 0.61
#    clf3 = GradientBoostingClassifier(n_estimators=100) # 0.61 
#    model = VotingClassifier(estimators=[('ab', clf1),('gdb', clf3)],
#                            voting='soft', weights=[1, 2])
    
    model.fit(Train_X, Train_Y)
    
    Train_Y_predict = model.predict(Train_X)
    Train_error = np.sum(Train_Y != Train_Y_predict) / len(Train_Y)
    print('Train error:', Train_error)
    
    Test_Y_predict = model.predict(Test_X)
    Test_error = np.sum(Test_Y != Test_Y_predict) / len(Test_Y)
    print('Test error:', Test_error)
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    main(args)
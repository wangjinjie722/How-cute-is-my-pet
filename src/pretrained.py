# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:38:38 2019

Get features of Vgg16 

@author: illya
"""
import os
os.chdir("Z://00_UCSD/ECE 271B/Project/") 

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import torch
import torch.nn as nn
import torchvision.models as models

#def vgg16_feature(x):
#    features = list(models.vgg16(pretrained = True).features)[:22]
#    feature = nn.ModuleList(features).eval() 
#    out = feature(x)
#    return out

class Vgg16_Feature(nn.Module):
    def __init__(self):
        super(Vgg16_Feature, self).__init__()
        features = list(models.vgg16(pretrained = True).features)[:24]
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
        return x

class Alexnet_Feature(nn.Module):
    def __init__(self):
        super(Alexnet_Feature, self).__init__()
        features = list(models.alexnet(pretrained = True).features)[:13]
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        for ii, model in enumerate(self.features):
            x = model(x)
        return x


if __name__ == '__main__':
    path = "data_organized_hist/"
    petid = ['0a83235b2','0a76a6cf3']
    x = np.zeros((2, 3, 128, 128))
    for i in range(2):
        try:
            x[i,0,:,:] = misc.imread(path+petid[i]+".png")[:,:,1].astype(float)
        except:
            continue
    model = Alexnet_Feature().double()
    out = model(torch.tensor(x))
    print(out.shape)
    out = out.detach().numpy()
#    out = out.reshape(2,512*8*8)
    
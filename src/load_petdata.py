#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:50:28 2019

Load pet dataset
Modified: read one image at a time
    
@author: illya
"""
import numpy as np
from scipy import misc
from torch.utils.data import Dataset

from characters import characters


#def imageread(petid, path):
#    n = len(petid)
#    images = np.random.randint(255, size=(n, 128, 128))
#    for i in range(n):
#        try:
#            images[i,:,:] = misc.imread(path+petid[i]+".png")[:,:,1]
#        except:
#            continue
#    return images

 
class PetDataset(Dataset):
    """ img: 128*128, label: 0~5 """
    def __init__(self, path_img, path_csv, istrain):
        self.path = path_img
        chara, petid, label = characters(path_csv)
        if istrain == 1:
            self.petid = petid[0:10000]
            self.label = label[0:10000]
            self.chara = chara[0:10000]
        else:
            self.petid = petid[10000:14993]
            self.label = label[10000:14993]
            self.chara = chara[10000:14993]
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        x = np.zeros((3, 128, 128))
        try:
            x[0,:,:] = misc.imread(self.path+self.petid[index]+".png")[:,:,0].astype(float)
            x[1,:,:] = misc.imread(self.path+self.petid[index]+".png")[:,:,1].astype(float)
            x[2,:,:] = misc.imread(self.path+self.petid[index]+".png")[:,:,2].astype(float)
        except:
            pass
        return (x, self.label[index])
    
    
if __name__ == "__main__":
    path_csv = "petfinder-adoption-prediction/train.csv"
    path_img = "data_organized_hist/"
    train_dataset = PetDataset(path_img, path_csv, 1)
    data, label = train_dataset[7]
    print(data.shape)
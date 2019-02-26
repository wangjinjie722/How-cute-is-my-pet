#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:34:16 2019

@author: ruoqi
"""
import numpy as np
from scipy import ndimage
from scipy import misc

def imageread(petid, path):
    n = len(petid)
    images = np.random.randint(255, size=(n, 128, 128))
    for i in range(n):
        try:
            images[i,:,:] = misc.imread(path+petid[i]+".png")[:,:,1]
        except:
            continue
    return images

if __name__ == "__main__":
    imageread()
    print(y.shape)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 02:39:40 2019

@author: ruoqi
"""

import numpy as np
import pandas as pd
import torch

def sigmoid(x):
    return 1/(1+np.exp(-x))

def normalize(x):
    return (x - x.mean()) / x.std()

def minmax(x):
    return (x.max()-x) / (x.max()-x.min())


def characters(path):
    raw = pd.read_csv(path)
    index = list(raw.columns.values)
    petid = raw['PetID'].values
    needindex=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    ndx = []
    ndata = raw[index[0]]
    for i in needindex:
        ndx.append(index[i])
        if i==0:
            ndata = raw[index[0]]
        else:
            ndata = pd.concat([ndata,raw[index[i]]],axis=1)
    umdata = ndata.values
    labels = raw[index[-1]].values
    data = normalize(umdata)

    return data, petid, labels
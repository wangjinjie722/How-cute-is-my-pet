#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 02:39:40 2019

@author: ruoqi
"""

import numpy as np
import pandas as pd
import torch
import sklearn

def sigmoid(x):
    return 1/(1+np.exp(-x))

def normalize(x):
    return (x - x.mean()) / x.std()

def minmax(x):
    return (x.max()-x) / (x.max()-x.min())


def characters(path,explore=False):
    if explore:
        ''' use train_explore'''
        raw = pd.read_csv(path)
        index = list(raw.columns.values)
        petid = raw['PetID'].values
        needindex=list(range(len(raw.columns.values)))[2:-2]
        ndx = []
        ndata = raw[index[0]]
        for i in needindex:
            ndx.append(index[i])
            if i==0:
                ndata = raw[index[0]]
            else:
                ndata = pd.concat([ndata,raw[index[i]]],axis=1)
        umdata = ndata.values
        umdata = umdata[:,1:]
        # set nan = 0
        where_are_nan = np.isnan(umdata)
        umdata[where_are_nan] = 0
        
        labels = raw[index[-1]].values
#        data = normalize(umdata)
        data = sklearn.preprocessing.normalize(umdata,axis=0)
#        data = umdata
    else:
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
#        data = umdata
    return data, petid, labels

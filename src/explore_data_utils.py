# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:04:57 2019

@author: xumw1
"""

import numpy as np
import pandas as pd

path = '../data/train.csv'
path_explore = '../data/train_explore.csv'

train_csv = pd.read_csv(path)
train_explore_csv = pd.read_csv(path_explore)

# add pet
train_explore_csv_mod = pd.concat([train_explore_csv, train_csv['PetID']],axis=1)
train_explore_csv_mod = pd.concat([train_explore_csv_mod, train_csv['AdoptionSpeed']],axis=1)

train_explore_csv_mod.to_csv('../data/train_explore_mod.csv')

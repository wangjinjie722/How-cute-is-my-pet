#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:34:50 2019

@author: ruoqi
"""
import numpy as np
def tensor2np(inputtensor, num):
    '''
    inputtensor: 输入tensor格式variable
    num: 每个tensor的list长度
    '''
    lenth = len(inputtensor)
    desvec = np.zeros((14993,64*num))
    for i in range(lenth):
            stack = np.zeros((num,64))
            for j in range(num):
                try:
                    stack[j] = inputtensor[i][j].detach().numpy().reshape(1,64)
                except:
                    continue
            desvec[i] = stack.reshape(1,64*num)
    return desvec


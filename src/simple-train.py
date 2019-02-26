#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:42:25 2019

@author: liyufei
"""

import csv
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import spacy
from collections import Counter


with open('/users/wangkai/Downloads/271project/train.csv',encoding='ISO-8859-1') as csvfile:
    reader=csv.reader(csvfile)
    description=[row[20] for row in reader]                 #all the description of pets 
description = description[1:]

#with open('/Users/liyufei/Desktop/train_description.txt', 'wb') as f:
#    for item in description:
 #       line = item +'\n'
#        f.write(line.encode('utf-8'))
        
with open('/users/wangkai/Downloads/271project/train.csv',encoding='utf-8') as csvfile:
    reader=csv.reader(csvfile)
    label=[row[23] for row in reader]                 #all the description of pets 
label = np.array(label)
label = label[1:]
#with open('/Users/liyufei/Desktop/train_label.txt', 'wb') as f:
 #   for item in label:
 #       line = item +'\n'
 #       f.write(line.encode('utf-8'))
        
adj_list = []
nlp = spacy.load('en')
for row in description:
    doc = nlp(row)
    for token in doc:
        if token.pos_ == 'ADJ':
            adj_list.append(token.text)

common_word = Counter(adj_list).most_common(200)

positive_word = []
for row in common_word:
    if row[0] not in ['grey','dan','basic','possible','only','dry','medium','same','my','your','stray','small','sure','compulsory','wet','last','due','serious','short','little','her','his','old','other','our','their','that','more','female','white','black','which','male','all','few','many','own','its','My','Her','human','short','pet','such','unable','His','poor','busy','bad','sad','previous','afraid','temporary','next','Their','hungry','alone','current','Serious','scared','older','Female','homeless','Our','past','less','normal','weak','main','All','Black','irresponsible','Its','Little']:
        positive_word.append(row[0])

embeds = nn.Embedding(len(positive_word), 64)  #  words in vocab, 64 dimensional embeddings
word_vector = []
for i in range(len(positive_word)):
    idx = torch.LongTensor([i])
    idx = Variable(idx)
    idx_embed = embeds(idx)
    word_vector.append(idx_embed)

##find positive in description
description_vector = []
for row in description:
    doc = nlp(row)
    row_vector = []
    for token in doc:
        if str(token) in positive_word:
            j = positive_word.index(str(token))
            row_vector.append(word_vector[j])
    description_vector.append(row_vector)

#1.1 one_hot method
#onehot_des = np.zeros((14993,64))
#for i in range(1,len(description_vector)):
#    po = len(description_vector[i])
#    if po != 0:
#        onehot_des[i-1][po-1] = 1
    
# threshold fetching words
threshold = 4
des_tensor = []
for i in description_vector:
    if len(i) <= 4:
        des_tensor.append(i)
    else:
        des_tensor.append(i[:4])
        
#train word with batch learning
def BatchLearning(eta,iteration,class_number,train_imgs,train_labels):
    train_imgs = np.array(train_imgs)
    [row,col] = train_imgs.shape
    train_imgs = np.column_stack((train_imgs,np.ones((row,1)))) #give bias
    train_labels_regular = np.zeros((row,class_number))
    for i in range(row):
        j = train_labels[i]
        train_labels_regular[i,int(float(j))] = 1 #regular t
    #initialize w
    mean = np.zeros((1,col+1))
    cov = np.eye(col+1)
    w = np.random.multivariate_normal(mean[0],cov,class_number)
    for i in range(iteration):
        a = w.dot(train_imgs.T)
        y = np.exp(a)/np.sum(np.exp(a),axis = 0)
        E_w = (train_labels_regular.T - y).dot(train_imgs)
        w += eta * E_w
    train_vector = train_imgs.dot(w.T)
    return w,train_vector

eta = 1.0e-3
iteration = 5000
class_number = 5
#[w,train_vector] = BatchLearning(eta,iteration,class_number,des_tensor,label)

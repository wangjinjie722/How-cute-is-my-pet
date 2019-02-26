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
import characters
import tensor2np


with open('/Users/wangkai/Downloads/271Project/train.csv',encoding='ISO-8859-1') as csvfile:
    reader=csv.reader(csvfile)
    description=[row[20] for row in reader]                 #all the description of pets
description = description[1:]


#with open('/Users/liyufei/Desktop/train_description.txt', 'wb') as f:
#    for item in description:
#        line = item +'\n'
#        f.write(line.encode('utf-8'))

with open('/Users/wangkai/Downloads/271Project/train.csv',encoding='utf-8') as csvfile:
    reader=csv.reader(csvfile)
    label=[row[23] for row in reader]                 #all the description of pets
label = np.array(label)
label = label[1:]
#with open('/Users/liyufei/Desktop/train_label.txt', 'wb') as f:
#    for item in label:
#        line = item +'\n'
#        f.write(line.encode('utf-8'))

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


#[w,train_vector] = BatchLearning(eta,iteration,class_number,des_tensor,label)

other_labels = characters('/Users/wangkai/Downloads/271Project/train.csv') # numpy

labels1 = other_labels[0]
labels1 = np.array(labels1)
labels_ID = other_labels[1]
other_vector = []
for j in range(16):
    labels = labels1[:,j]
    embeds = nn.Embedding(len(labels), 64)  #  words in vocab, 64 dimensional embeddings
    other_vector.append([])
    for i in range(len(labels)):
        idx = torch.LongTensor([i])
        idx = Variable(idx)
        idx_embed = embeds(idx)
        other_vector[j].append(idx_embed)

npath = '/Users/wangkai/Downloads/271Project/train.csv'

other = np.load(npath)

other = other.T

nother = np.zeros((14993,16,64))

for i in range(16):

    nother[:,i,:] = tensor2np(other[:,i].T,1)

    others = nother.reshape(14993, 16 * 64)  #

description_res = tensor2np(des_tensor,4)
otherlabels_res = others
np.save("description_res.npy", description_res);
np.save("otherlabels_res.npy", otherlabels_res);



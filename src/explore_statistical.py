# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:13:56 2019

@author: xumw1
"""
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import os

from characters import characters
from sklearn.externals.six.moves import zip
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def use_data():
    '''baseline'''
    print('-------baseline--------')
    path = '../data/train.csv'
    
    data, petid, labels = characters(path, explore=False)
    X = data
    y = labels
    
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state = 0)
    print("X_train created, shape = " + str(X_train.shape))
    print("y_train created, shape = " + str(y_train.shape))
    
    print("X_test created, shape = " + str(X_test.shape))
    print("y_test created, shape = " + str(y_test.shape))
    # compare between models
    #print('SVM')
    #clf = SVC(gamma='auto')
    #clf.fit(X_train, y_train) 
    #print("Training set score: {:.3f}".format(clf.score(X_train, y_train)))
    #print("Test set score: {:.3f}".format(clf.score(X_test, y_test)))
    
    print('AdaBoost')
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    print("Training set score: {:.3f}".format(ada.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(ada.score(X_test, y_test)))
    
#    print('AdaBoosted Decision Trees')
#    bdt_real = AdaBoostClassifier(
#        DecisionTreeClassifier(max_depth=2),
#        n_estimators=600,
#        learning_rate=1)
#    
#    bdt_discrete = AdaBoostClassifier(
#        DecisionTreeClassifier(max_depth=2),
#        n_estimators=600,
#        learning_rate=1.5,
#        algorithm="SAMME")
#    
#    bdt_real.fit(X_train, y_train)
#    bdt_discrete.fit(X_train, y_train)
#    
#    real_test_errors = []
#    discrete_test_errors = []
#    
#    for real_test_predict, discrete_train_predict in zip(
#            bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
#        real_test_errors.append(
#            1. - accuracy_score(real_test_predict, y_test))
#        discrete_test_errors.append(
#            1. - accuracy_score(discrete_train_predict, y_test))
#    print("Test set score: {:.3f}".format(1-real_test_errors[-1]))
    
    '''prediction'''
    print('prediction')
    ada.fit(X, y)
    path_test = '../data/test.csv'
    data_predict, _, _ = characters(path_test, explore=False)
    
    X_predict = data_predict
    predict = ada.predict(X_predict)
    
    return predict

def use_explored_data():
    '''explored data'''
    print('-------revised--------')
    path = '../data/train_explore_mod.csv'
    
    data, petid, labels = characters(path, explore=True)
    X = data
    y = labels
    
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state = 42)
    print("X_train created, shape = " + str(X_train.shape))
    print("y_train created, shape = " + str(y_train.shape))
    
    print("X_test created, shape = " + str(X_test.shape))
    print("y_test created, shape = " + str(y_test.shape))
    
    # compare between models
    #print('SVM')
    #clf = SVC(gamma='auto')
    #clf.fit(X_train, y_train) 
    #print("Training set score: {:.3f}".format(clf.score(X_train, y_train)))
    #print("Test set score: {:.3f}".format(clf.score(X_test, y_test)))
    
    print('AdaBoost')
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    print("Training set score: {:.3f}".format(ada.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(ada.score(X_test, y_test)))
    
    '''prediction'''
    print('prediction')
    ada.fit(X, y)
    path_test = '../data/test.csv'
    data_predict, _, _ = characters(path_test, explore=True)
    
    X_predict = data_predict
    predict = ada.predict(X_predict)
    
    return predict

if __name__ == '__main__':
    prediction0 = use_data()
#    prediction1 = use_explored_data()
#
    sub = pd.read_csv(os.path.join('../input/test/sample_submission.csv'))
    
    submission0 = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction0]})
    submission0.head()
    submission0.to_csv('../submission_tableonly.csv', index=False)
#    
#    
#    submission1 = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction1]})
#    submission1.head()
#    submission1.to_csv('../submission_exploretableonly.csv', index=False)

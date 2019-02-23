# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:16:46 2019

@author: xumw1

How-cute-is-my-pet
Image Multithreading Preprocess
"""


import cv2, os
import glob
import numpy as np
import time

from multiprocessing import Pool
from catface_detector import *


def catface_generate_multiprocess(folder, data):
    '''
    read images from folder
    export catface/dogface images to folder_save
    '''
    imlist = glob.glob(folder + "\*.jpg")
    #一般字典序排列,reverse = True 降序,reverse = False 升序（默认）
    imlist.sort()

    if data == 'train':
        # state checker
        t_start = time.clock()
        # loop
        pool = Pool()
        pool.map(catface_helper_train, imlist)
        pool.close()
        pool.join()
        print('runtime = ', time.clock() - t_start)
    if data == 'test':
        # state checker
        t_start = time.clock()
        # loop
        pool = Pool()
        pool.map(catface_helper_test, imlist)
        pool.close()
        pool.join()
        print('runtime = ', time.clock() - t_start)

def catface_helper_train(item):
    # helper function for catface_generate_multiprocess
    # multiprocess compatible
    detector = catface_detector()
    folder_save = os.path.abspath('.') + '\\data'
    img = cv2.imread(item)
    try:
        img_catface = detector.get_catface(img, offset=20)
        name_save = folder_save + '\\' + item.split('\\')[-1]
        cv2.imwrite(name_save, img_catface)
        return None
    except:
        return None

def catface_helper_test(item):
    # helper function for catface_generate_multiprocess
    # multiprocess compatible
    detector = catface_detector()
    folder_save = os.path.abspath('.') + '\\testdata'
    img = cv2.imread(item)
    try:
        img_catface = detector.get_catface(img, offset=20)
        name_save = folder_save + '\\' + item.split('\\')[-1]
        cv2.imwrite(name_save, img_catface)
        return None
    except:
        return None

def grayhist_generate_multiprocess(folder, data):
    '''
    read images from folder
    export 128*128 grayscale histogram_regularized images to folder_save
    '''
    imlist = glob.glob(folder + "\*.jpg")
    #一般字典序排列,reverse = True 降序,reverse = False 升序（默认）
    imlist.sort()

    if data == 'train':
        # state checker
        t_start = time.clock()
        # loop
        pool = Pool()
        pool.map(grayhist_helper_train, imlist)
        pool.close()
        pool.join()
        print('runtime = ', time.clock() - t_start)
    if data == 'test':
        # state checker
        t_start = time.clock()
        # loop
        pool = Pool()
        pool.map(grayhist_helper_test, imlist)
        pool.close()
        pool.join()
        print('runtime = ', time.clock() - t_start)


def grayhist_helper_train(item):
    # helper function for grayhist_generate_multiprocess
    # multiprocess compatible
    folder_save = os.path.abspath('.') + '\\data_regularize_hist'
    img = cv2.imread(item)
    try:
        img_resize = cv2.resize(img, (128, 128))  # resize
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)  # grayscale
        img_hist = cv2.equalizeHist(img_resize)  # 均衡化
        # save
        name_save = folder_save + '\\' + item.split('\\')[-1]
        cv2.imwrite(name_save, img_hist)
        return None
    except:
        return None


def grayhist_helper_test(item):
    # helper function for grayhist_generate_multiprocess
    # multiprocess compatible
    folder_save = os.path.abspath('.') + '\\testdata_regularize_hist'
    img = cv2.imread(item)
    try:
        img_resize = cv2.resize(img, (128, 128))  # resize
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)  # grayscale
        img_hist = cv2.equalizeHist(img_resize)  # 均衡化
        # save
        name_save = folder_save + '\\' + item.split('\\')[-1]
        cv2.imwrite(name_save, img_hist)
        return None
    except:
        return None


def organize_data_multiprocess(folder, data):
    '''
    read images from folder
    save one image for each petID to folder_save, format in png
    '''
    imlist = glob.glob(folder + "\*.jpg")

    if data == 'train':
        # state checker
        t_start = time.clock()
        pool = Pool()
        pool.map(organize_helper_train, imlist)
        pool.close()
        pool.join()
        print('runtime = ', time.clock() - t_start)
    if data == 'test':
        # state checker
        t_start = time.clock()
        pool = Pool()
        pool.map(organize_helper_test, imlist)
        pool.close()
        pool.join()
        print('runtime = ', time.clock() - t_start)


def organize_helper_train(item):
    # helper function for organize_data_multiprocess
    # multiprocess compatible
    img_2 = cv2.imread(item)
    '''
    resize and grayscale
    '''
    folder_save_2 = os.path.abspath('.') + '\\data_organized_hist'
    # save
    item2 = item.split('\\')[-1]
    item2 = item2.split('-')[0]
    item2 = item2 + '.png'

    name_save_2 = folder_save_2 + '\\' + item2
    cv2.imwrite(name_save_2, img_2)
    return None


def organize_helper_test(item):
    # helper function for organize_data_multiprocess
    # multiprocess compatible
    img_2 = cv2.imread(item)
    '''
    resize and grayscale
    '''
    folder_save_2 = os.path.abspath('.') + '\\testdata_organized_hist'
    # save
    item2 = item.split('\\')[-1]
    item2 = item2.split('-')[0]
    item2 = item2 + '.png'

    name_save_2 = folder_save_2 + '\\' + item2
    cv2.imwrite(name_save_2, img_2)
    return None


def preprocess(traindata=False,
               testdata=False,
               detect=False,
               grayscale=False,
               organize=False):
    '''
    folders are in relative path, makesure to create them before running the program
    traindata: process traindata
    testdata: process testdata
    detect: use cat-face detector to generate face imgs
    grayscale: convert face imgs to grayscale, then histogram regularize
    organize: keep only one img for each petID, save as .png
    '''
    t_start_all = time.clock()
    if traindata:
        print('===============Preprocessing train data==================')
        if detect:
            # detect catface
            print('---------------Detecting---------------')
            folder = os.path.abspath(
                '..') + r'\petfinder-adoption-prediction\train_images'
            folder_save = os.path.abspath('.') + '\\data'

            catface_generate_multiprocess(folder, 'train')

            time.sleep(1)
        # grayscale and normalize
        if grayscale:
            print('--------------Normalizing--------------')
            print('using multiprocess')
            folder = os.path.abspath('.') + '\\data'

            grayhist_generate_multiprocess(folder, 'train')

            time.sleep(1)
        if organize:
            # remove redundancy
            print('--------------Organizing---------------')
            print('using multiprocess')
            folder = os.path.abspath('.') + '\\data_regularize_hist'
            organize_data_multiprocess(folder, 'train')
            time.sleep(1)
    if testdata:
        print('===============Preprocessing test data==================')
        if detect:
            # detect catface
            print('---------------Detecting---------------')
            folder = os.path.abspath(
                '..') + r'\petfinder-adoption-prediction\test_images'
            folder_save = os.path.abspath('.') + '\\testdata'

            catface_generate_multiprocess(folder, 'test')

            time.sleep(1)
        if grayscale:
            # grayscale and normalize
            print('--------------Normalizing--------------')
            print('using multiprocess')
            folder = os.path.abspath('.') + '\\testdata'
            folder_save = os.path.abspath('.') + '\\testdata_regularize_hist'

            grayhist_generate_multiprocess(folder, 'test')

            time.sleep(1)
        if organize:
            # remove redundancy
            print('--------------Organizing---------------')
            print('using multiprocess')
            folder = os.path.abspath('.') + '\\testdata_regularize_hist'
            organize_data_multiprocess(folder, 'test')
            time.sleep(1)
    print('preprocess finish')
    print('runtime = ', time.clock() - t_start_all)


if __name__ == '__main__':
    preprocess(traindata=False, testdata=True, detect=True, grayscale=False)

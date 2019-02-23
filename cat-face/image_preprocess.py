import cv2, os
import glob
import numpy as np
import time
from tqdm import tqdm

from catface_detector import *


def catface_generate(folder, folder_save):
    '''
    read images from folder
    export catface/dogface images to folder_save
    '''
    detector = catface_detector()
    imlist = glob.glob(folder + "\*.jpg")
    #一般字典序排列,reverse = True 降序,reverse = False 升序（默认）
    imlist.sort()

    # state checker
    pbar = tqdm(total=len(imlist))  # initialize pbar
    t_start = time.clock()
    # loop
    for item in imlist:
        img = cv2.imread(item)
        pbar.update(1)
        try:
            img_catface = detector.get_catface(img, offset=20)
            name_save = folder_save + '\\' + item.split('\\')[-1]
            cv2.imwrite(name_save, img_catface)
        except:
            continue
    pbar.close()
    print('runtime = ', time.clock() - t_start)


def grayhist_generate(folder, folder_save, size=128):
    '''
    read images from folder
    export 128*128 grayscale histogram_regularized images to folder_save
    '''
    imlist = glob.glob(folder + "\*.jpg")
    #一般字典序排列,reverse = True 降序,reverse = False 升序（默认）
    imlist.sort()

    # state checker
    pbar = tqdm(total=len(imlist))  # initialize pbar
    t_start = time.clock()
    # loop
    for item in imlist:
        img = cv2.imread(item)
        '''
        resize and grayscale
        '''
        try:
            img_resize = cv2.resize(img, (size, size))  # resize
            img_resize = cv2.cvtColor(img_resize,
                                      cv2.COLOR_BGR2GRAY)  # grayscale
            img_hist = cv2.equalizeHist(img_resize)  # 均衡化

            pbar.update(1)

            # save
            name_save = folder_save + '\\' + item.split('\\')[-1]
            cv2.imwrite(name_save, img_hist)
        except:
            continue
    pbar.close
    print('runtime = ', time.clock() - t_start)


def regularize_data(folder, folder_save):
    '''
    read images from folder
    save one image for each petID to folder_save, format in png
    '''
    folder_load_2 = folder
    folder_save_2 = folder_save

    imlist2 = glob.glob(folder_load_2 + "\*.jpg")

    # state checker
    pbar = tqdm(total=len(imlist2))  # initialize pbar
    t_start = time.clock()
    # loop
    for item in imlist2:
        img_2 = cv2.imread(item)
        '''
        resize and grayscale
        '''
        pbar.update(1)

        # save
        item2 = item.split('\\')[-1]
        item2 = item2.split('-')[0]
        item2 = item2 + '.png'

        name_save_2 = folder_save_2 + '\\' + item2
        cv2.imwrite(name_save_2, img_2)

    pbar.close
    print('runtime = ', time.clock() - t_start)


def preprocess(traindata=False, testdata=False, detect=False, grayscale=False, organize=False):
    '''
    folders are in relative path
    '''
    t_start_all = time.clock()
    if traindata:
        if detect:
            print('===============Preprocessing train data==================')

            # detect catface
            print('---------------Detecting---------------')
            folder = os.path.abspath(
                '..') + r'\petfinder-adoption-prediction\train_images'
            folder_save = os.path.abspath('.') + '\\data'

            catface_generate(folder, folder_save)

            time.sleep(1)
        if grayscale:
            # grayscale and normalize
            print('--------------Normalizing--------------')
            folder = os.path.abspath('.') + '\\data'
            folder_save = os.path.abspath('.') + '\\data_regularize_hist'

            grayhist_generate(folder,folder_save)

            time.sleep(1)
        if organize:
            # remove redundancy
            print('--------------Organizing---------------')
            folder = os.path.abspath('.') + '\\data_regularize_hist'
            folder_save = os.path.abspath('.') + '\\data_organized_hist'

            regularize_data(folder, folder_save)
            time.sleep(1)
    if testdata:
        if detect:
            print('===============Preprocessing test data==================')
            # detect catface
            print('---------------Detecting---------------')
            folder = os.path.abspath(
                '..') + r'\petfinder-adoption-prediction\test_images'
            folder_save = os.path.abspath('.') + '\\testdata'

            catface_generate(folder, folder_save)

            time.sleep(1)
        if grayscale:
            # grayscale and normalize
            print('--------------Normalizing--------------')
            folder = os.path.abspath('.') + '\\testdata'
            folder_save = os.path.abspath('.') + '\\testdata_regularize_hist'

            grayhist_generate(folder,folder_save)

            time.sleep(1)
        if organize:
            # remove redundancy
            print('--------------Organizing---------------')
            folder = os.path.abspath('.') + '\\testdata_regularize_hist'
            folder_save = os.path.abspath('.') + '\\testdata_organized_hist'

            regularize_data(folder, folder_save)
            time.sleep(1)
    print('preprocess finish')
    print('runtime = ', time.clock() - t_start_all)

if __name__ == '__main__':
    preprocess(traindata=True, testdata=False, organize=True)

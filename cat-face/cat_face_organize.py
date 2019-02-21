# catface regularize
import cv2, os
import glob
import numpy as np
import time
from tqdm import tqdm

if __name__ == "__main__":
    folder_load_1 = r"F:\Projects\ECE_271B\cat-face\data_regularize"
    folder_load_2 = r"F:\Projects\ECE_271B\cat-face\data_regularize_hist"

    folder_save_1 = r"F:\Projects\ECE_271B\cat-face\data_organized"
    folder_save_2 = r"F:\Projects\ECE_271B\cat-face\data_organized_hist"

    imlist1 = glob.glob(folder_load_1 + "\*.jpg")
    imlist2 = glob.glob(folder_load_2 + "\*.jpg")

    # state checker
    pbar = tqdm(total=len(imlist1))  # initialize pbar
    t_start = time.clock()
    # loop
    for item in imlist1:
        img_1 = cv2.imread(item)
        '''
        resize and grayscale
        '''
        pbar.update(1)

        # save
        item1 = item.split('\\')[-1]
        item1 = item1.split('-')[0]
        item1 = item1 + '.png'

        name_save_1 = folder_save_1 + '\\' + item1
        cv2.imwrite(name_save_1, img_1)

    pbar.close
    print('runtime = ', time.clock() - t_start)

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
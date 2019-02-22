import cv2, os, sys
import glob
import numpy as np
import time
from tqdm import tqdm


class catface_detector(object):
    def __init__(self):
        # 加载猫脸检测器
        catPath = os.path.abspath('.') + r"\haarcascade_frontalcatface.xml"
        if os.path.exists(catPath) == False:
            sys.exit('Cascade does not exist!')
        #catPath = r"F:\Projects\ECE_271B\How-cute-is-my-pet\cat-face\haarcascade_frontalcatface.xml"
        self.faceCascade = cv2.CascadeClassifier(catPath)

    def get_catface(self, img, offset=0):
        '''
        type img: img, input image with cat
        type offset: float, offset size of cat face
        rtype img_catface: img, img of catface
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 猫脸检测
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.03,  # default 1.02 set 1.018
            minNeighbors=3,  # min number of detected bounding boxes default 3 set 2
            minSize=(70, 70),  # min size of bounding box  default 150, 150
            flags=cv2.CASCADE_SCALE_IMAGE)
        # 框出猫脸并加上文字说明
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.putText(img, 'Cat', (x, y - 7), 3, 1.2, (0, 255, 0), 2,
        #                 cv2.LINE_AA)
        # print("face coordinate = ", faces)

        # choose the box with larges area
        area = []
        for (x, y, w, h) in faces:
            area.append(w * h)
        try:
            ind = np.argmax(area)
        except:
            raise ValueError
        (x, y, w, h) = faces[ind]
        ''' more than one faces: what to do?
            no face: what to do
            '''

        # 显示图片并保存
        # cv2.imshow('Cat?', img)
        # cv2.waitKey(0)

        # 截取猫脸
        try:
            img_catface = img[y - offset:y + h + offset, x - offset:x + w +
                              offset]
            # cv2.imshow("catface", img_catface)
            # cv2.waitKey(0)
        # 未检测到猫脸
        except:
            raise ValueError

        return img_catface


# if __name__ == "__main__":
#     detector = catface_detector()
#     folder = r"F:\Projects\ECE_271B\How-cute-is-my-pet\cat-face"  #test folder
#     #folder = r"F:\Projects\ECE_271B\petfinder-adoption-prediction\train_images"
#     #folder_save = r"F:\Projects\ECE_271B\cat-face\data"
#     folder_save = r"F:\Projects\ECE_271B\How-cute-is-my-pet\cat-face\testlog"

#     # for filename in os.listdir(folder):
#     #     img = cv2.imread(os.path.join(folder,filename))
#     #     detector = catface_detector()
#     #     detector.get_catface(img)

#     imlist = glob.glob(folder + "\*.jpg")
#     print(imlist)
#     #一般字典序排列,reverse = True 降序,reverse = False 升序（默认）
#     imlist.sort()

#     # state checker
#     pbar = tqdm(total=len(imlist))  # initialize pbar
#     t_start = time.clock()
#     # loop
#     for item in imlist:
#         img = cv2.imread(item)
#         pbar.update(1)
#         try:
#             img_catface = detector.get_catface(img, offset=20)
#         #print(item)
#         # if img_catface != None:
#         #     print('save')
#         #     cv2.imshow(img_catface)
#             name_save = folder_save + '\\' + item.split('\\')[-1]

#             cv2.imwrite(name_save, img_catface)
#         except:
#             print('no img')
#             continue
#     pbar.close()
#     print('runtime = ', time.clock() - t_start)

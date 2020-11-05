import numpy as np
import os
import cv2

path = '../sub_img'

filenames = os.listdir(path)
imgs_npy = []
for filename in filenames:
    img = cv2.imread(path+os.sep+filename)
    imgs_npy.append(img)
imgs_npy = np.array(imgs_npy)
np.save('sub_img.npy', imgs_npy)
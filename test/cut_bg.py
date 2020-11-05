import cv2
import os
cnt = 0
path = '../bg'
filenames = os.listdir(path)
for filename in filenames:
    img = cv2.imread('../bg/'+filename)
    h, w = img.shape[:2]
    i = 0
    j = 0
    sub_img = None
    while i < h-28:
        while j < w-28:
            sub_img = img[i:i+28, j:j+28, :]
            cv2.imwrite("../sub_img/"+str(cnt)+".jpg", sub_img)
            cnt += 1
            j += 28
        i += 28
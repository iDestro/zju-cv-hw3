import numpy as np
import cv2


img = np.ones([100, 100, 3])*255
cv2.imshow('img', img)
cv2.waitKey(0)
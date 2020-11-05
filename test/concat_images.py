from dateset import BgMNIST
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

train_set = np.load('train_set_data_modify.npy')
sub = train_set[:100]
print(sub[0].shape)


temp_1 = None
for i in range(10):
    temp_0 = None
    for j in range(10):
        if temp_0 is None:
            temp_0 = sub[i*10+j]
        else:
            temp_0 = np.concatenate([temp_0, sub[i*10+j]], axis=1)

    if temp_1 is None:
        temp_1 = temp_0
    else:
        temp_1 = np.concatenate([temp_1, temp_0], axis=0)

print(temp_1.shape)

cv2.imshow('img', temp_1)
cv2.imwrite('../test_images/rgb_multi.jpg', temp_1)
cv2.waitKey(0)
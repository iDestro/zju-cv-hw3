import torchvision
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


train_set = torchvision.datasets.MNIST('./',
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())
test_set = torchvision.datasets.MNIST('./',
                                      train=False,
                                      download=True,
                                      transform=transforms.ToTensor())

print(train_set.data.shape)
print(train_set.targets.shape)
bg_imgs = np.load('./sub_img.npy')

indices = np.arange(bg_imgs.shape[0])

train_set_data_modify = []
train_set_target_binary = []
train_set_target_multi = []
cnt = 0
for x, y in zip(train_set.data, train_set.targets):
    x = x.numpy()
    y = y.numpy()
    if cnt == 1000:
        plt.imshow(x)
        plt.show()
    cnt += 1
    temp = x.copy()
    idx = np.random.choice(indices)
    bg_img = bg_imgs[idx].copy()
    rgb_x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    rgb_x[x == 0] = np.array([0, 0, 0])
    rgb_x[x > 0] = np.array([255, 255, 255])
    bg_img[x > 0] = np.array([0, 0, 0])
    bg_img = bg_img + rgb_x
    train_set_data_modify.append(bg_img.copy())
    x[x > 0] = 1
    x[x == 0] = 0
    train_set_target_binary.append(x.copy())
    # cv2.imshow('binary', x)
    x[temp == 0] = 10
    x[temp > 0] = y
    if np.sum(x) == 7840:
        plt.imshow(x)
        plt.show()
    train_set_target_multi.append(x.copy())
    # cv2.imshow('multi', x)


train_set_data_modify = np.array(train_set_data_modify)
train_set_target_binary = np.array(train_set_target_binary)
train_set_target_multi = np.array(train_set_target_multi)

np.save('train_set_data_modify.npy', train_set_data_modify)
np.save('train_set_target_binary.npy', train_set_target_binary)
np.save('train_set_target_multi.npy', train_set_target_multi)

test_set_data_modify = []
test_set_target_binary = []
test_set_target_multi = []


for x, y in zip(test_set.data, test_set.targets):
    x = x.numpy()
    y = y.numpy()
    temp = x.copy()
    idx = np.random.choice(indices)
    bg_img = bg_imgs[idx].copy()
    rgb_x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    rgb_x[x == 0] = np.array([0, 0, 0])
    rgb_x[x > 0] = np.array([255, 255, 255])
    bg_img[x > 0] = np.array([0, 0, 0])
    bg_img = bg_img + rgb_x
    # cv2.imshow('bg_added_img', bg_img)
    test_set_data_modify.append(bg_img.copy())
    x[x > 0] = 1
    x[x == 0] = 0
    test_set_target_binary.append(x.copy())
    # cv2.imshow('binary', x)
    x[temp == 0] = 10
    x[temp > 0] = y
    if np.sum(x) == 7840:
        plt.imshow(x)
        plt.show()
    test_set_target_multi.append(x.copy())
    # cv2.imshow('multi', x)


test_set_data_modify = np.array(test_set_data_modify)
test_set_target_binary = np.array(test_set_target_binary)
test_set_target_multi = np.array(test_set_target_multi)

np.save('test_set_data_modify.npy', test_set_data_modify)
np.save('test_set_target_binary.npy', test_set_target_binary)
np.save('test_set_target_multi.npy', test_set_target_multi)

cv2.waitKey(0)
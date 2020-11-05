import matplotlib.pyplot as plt
import numpy as np


imgs = np.load('./train_set_data_modify.npy')
targets = np.load('./train_set_target_multi.npy')

imgs = imgs[30000:30010]
targets = targets[30000:30010]

for i in range(imgs.shape[0]):
    plt.imshow(imgs[i])
    plt.show()
    plt.imshow(targets[i])
    plt.show()
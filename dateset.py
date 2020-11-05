from torch.utils.data import Dataset
import wget
import os
import torch
import numpy as np
import cv2


class BgMNIST(Dataset):
    def __init__(self, training, target_type='binary', color='rgb'):
        self.root = './BgMNIST'
        self.train_resources = {
            'train_set_data_modify.npy': 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/train_set_data_modify.npy',
            'train_set_target_binary.npy': 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/train_set_target_binary.npy',
            'train_set_target_multi.npy': 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/train_set_target_multi.npy'
        }
        self.test_resources = {
            'test_set_data_modify.npy': 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/test_set_data_modify.npy',
            'test_set_target_binary.npy': 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/test_set_target_binary.npy',
            'test_set_target_multi.npy': 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/test_set_target_multi.npy'
        }
        self.data, self.targets = self.download(training, target_type)
        if color == 'gray':
            gray_data = []
            for i in range(self.data.shape[0]):
                img = np.transpose(self.data[i].numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = torch.from_numpy(img)
                img = torch.unsqueeze(img, dim=0)
                gray_data.append(img)
            gray_data = torch.cat(gray_data, dim=0)
            self.data = torch.unsqueeze(gray_data, dim=1)

    def __len__(self):
        data = self.data
        return data.size()[0]

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def download(self, training, target_type):
        data = None
        targets = None
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        resources = self.train_resources if training else self.test_resources
        for filename, url in resources.items():
            is_exist = os.path.exists(self.root + os.sep + filename)
            if 'data' in filename or target_type in filename:
                if not is_exist:
                    wget.download(url, self.root + os.sep + filename)
                    print(filename+" is finished!")
                else:
                    print(filename + " is exist!")
                if 'data' in filename:
                    data = np.load(self.root + os.sep + filename)
                    data = data.transpose([0, 3, 1, 2])
                    data = torch.Tensor(data)
                if target_type in filename:
                    targets = np.load(self.root + os.sep + filename)
                    targets = torch.LongTensor(targets)
        return data, targets

import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Config:
    def __init__(self):
        self.model_name = 'Net'
        self.learn_rate = 0.02
        self.num_epochs = 5
        self.batch_size = 128
        self.dropout = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filter_nums = 64


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.filter_nums = 88
        # (28, 28) -> (14, 14)
        self.conv1 = nn.Sequential(nn.Conv2d(1, self.filter_nums, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(self.filter_nums),
                                   nn.MaxPool2d(2, 2))
        # (14, 14) -> (7, 7)
        self.conv2 = nn.Sequential(nn.Conv2d(self.filter_nums, self.filter_nums*2, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(self.filter_nums*2),
                                   nn.MaxPool2d(2, 2))
        # (7, 3) -> (3, 3)
        self.conv3 = nn.Sequential(nn.Conv2d(self.filter_nums*2, self.filter_nums*4, 2, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(self.filter_nums*4),
                                   nn.MaxPool2d(2, 2))
        # Channel: 88 -> 11
        self.aggr1 = nn.Conv2d(self.filter_nums, 11, 1)
        # Channel: 88*2 -> 11
        self.aggr2 = nn.Conv2d(self.filter_nums*2, 11, 1)
        # Channel: 88*4 -> 11
        self.aggr3 = nn.Conv2d(self.filter_nums*4, 11, 1)

    def forward(self, x):
        s1 = self.conv1(x)
        s2 = self.conv2(s1)
        s3 = self.conv3(s2)
        s1 = self.aggr1(s1)
        s2 = self.aggr2(s2)
        s3 = self.aggr3(s3)
        s1 = F.interpolate(s1, scale_factor=x.size(3)//s1.size(3))
        s2 = F.interpolate(s2, scale_factor=x.size(3)//s2.size(3))
        s3 = F.interpolate(s3, scale_factor=x.size(3)//s3.size(3))
        x = s1+s2+s3
        return x


if __name__ == '__main__':
    dummy_input = torch.rand(1, 3, 28, 28)  # 假设输入13张1*28*28的图片
    model = Net()
    out = model(dummy_input)

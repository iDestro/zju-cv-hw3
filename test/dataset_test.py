from dateset import BgMNIST
import torch
from torch.utils.data import DataLoader


train_set = BgMNIST(training=False, target_type='multi')




train_iter = DataLoader(train_set, shuffle=True, batch_size=10)

for x, y in train_iter:
    print(x.size(), y.size())


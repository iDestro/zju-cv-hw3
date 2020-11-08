import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from dateset import BgMNIST
from model import Config, Net
from train import train
from evaluate import evaluate
import numpy as np
import argparse

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--flag', '-f', action='store_true', help='Is the number of convolution kernels halved?')
# parser.add_argument('--dropout_rate', '-d', type=float, default=0.0, help='Please set a dropout rate')
# args = parser.parse_args()

if __name__ == '__main__':
    train_set = BgMNIST(training=True, target_type='multi')
    test_set = BgMNIST(training=False, target_type='multi')
    config = Config()
    train_iter = DataLoader(train_set,
                            shuffle=True,
                            batch_size=config.batch_size)
    model = Net()
    # train(model, config, train_iter)
    evaluate(test_set, config)

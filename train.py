from datetime import timedelta
import torch
import numpy as np
import time
from model import Net
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt


def accuracy_score(y, y_pred):
    y_pred = torch.argmax(y_pred, dim=-3)
    ac = torch.sum(y == y_pred) / torch.Tensor([torch.numel(y_pred)]).cuda()*100
    return ac.cpu().item()


def IoU_score(y, y_pred):
    pass


def train(model, config, train_iter, test_iter):
    epoch_start_time = time.time()
    model.to(config.device)
    total_batch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(config.num_epochs):
        for i, data in enumerate(train_iter):
            x, y = data[0].to(config.device), data[1].to(config.device)
            y_pred = model(x)
            batch_loss = loss(y_pred, y)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                train_acc = accuracy_score(y, y_pred)
                train_loss = batch_loss.item()
                msg = 'Time: {},  Train Loss: {},  Train Acc: {}'
                time_diff = timedelta(seconds=int(round(time.time()-epoch_start_time)))
                print(msg.format(time_diff, train_loss, train_acc))
            total_batch += 1

    model.eval()
    torch.save(model.state_dict(), 'model_gray_multi.pkl')


def test():
    model = Net()
    model.load_state_dict(torch.load('./model_gray_multi.pkl'))
    model.eval()
    imgs = np.load('./BgMNIST/train_set_data_modify.npy')
    img = imgs[5000]
    img = cv2.imread('test_images/gray_multi.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(img)
    plt.show()
    img = torch.tensor(img, dtype=torch.float)
    img = torch.unsqueeze(img, dim=0)
    img = torch.unsqueeze(img, dim=0)
    print(img.size())
    out = model(img)
    y_pred = torch.argmax(out, dim=-3)
    y_pred = y_pred.cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    plt.imshow(y_pred)
    plt.show()
    # cv2.imshow('img2', y_pred)
    # cv2.waitKey(0)


if __name__ == '__main__':
    test()

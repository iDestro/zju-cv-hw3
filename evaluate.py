from model import Net
import torch
from utils import accuracy_score, miou
import numpy as np
from sklearn import metrics
from visualize import visualize
import cv2


def evaluate(test_set, config, gray=False):
    model = Net().cuda()
    model.load_state_dict(torch.load('./saved_dict/model_xjl_multi.pkl' if config.target_type =='multi' else './saved_dict/model_binary_multi.pkl'))
    model.eval()
    acc = 0
    labels_all = np.array([], dtype=int)
    predicts_all = np.array([], dtype=int)
    cnt = 0
    images = []
    predict_images = []
    with torch.no_grad():
        for x, y in test_set:
            if gray:
                images.append(np.squeeze(x.numpy(), axis=0))
            else:
                images.append(np.transpose(x.numpy(), axes=[1, 2, 0]))
            x = torch.unsqueeze(x, 0).cuda()
            y = torch.unsqueeze(y, 0).cuda()
            y_pred = model(x)
            predict = torch.argmax(y_pred, dim=-3).long().cpu().numpy()
            if gray:
                print(predict.shape)
                predict_images.append(np.squeeze(predict, axis=0))
            else:
                predict_images.append(cv2.cvtColor(np.squeeze(predict, axis=0).astype(np.uint8), cv2.COLOR_GRAY2RGB))
            labels = y.flatten().cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predicts_all = np.append(predicts_all, predict.flatten())
            acc += accuracy_score(y, y_pred)
            if cnt == 100:
                break
            cnt += 1
    visualize(images, predict_images, config)
    confusion = metrics.confusion_matrix(labels_all, predicts_all)
    print(acc / len(test_set))
    print(confusion)
    print(miou(confusion))


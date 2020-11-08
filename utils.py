import torch
import numpy as np


def accuracy_score(y, y_pred):
    y_pred = torch.argmax(y_pred, dim=-3)
    ac = torch.sum(y == y_pred) / torch.Tensor([torch.numel(y_pred)]).cuda()*100
    return ac.cpu().item()


def miou(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU
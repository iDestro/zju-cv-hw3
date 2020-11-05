import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PixelCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        # (N, C, H, W)
        input = F.log_softmax(input, dim=-3)
        print(input)
        # print(input)
        # print(target)
        one_hot = torch.zeros(input.shape).scatter_(1, torch.unsqueeze(target, dim=1), 1)
        # print(one_hot)
        print(torch.unsqueeze(target, dim=1))
        print(torch.zeros(input.shape))
        print(one_hot)
        print(torch.unsqueeze(target, dim=-1).size())
        print(input.size())
        print(one_hot.size())
        loss = input*one_hot
        loss = torch.sum(loss) / (input.shape[0]*input.shape[2]*input.shape[3])
        return -loss


if __name__ == '__main__':
    dummy_input = torch.rand(1, 4, 2, 2)  # 假设输入1张1*28*28的图片
    # print(dummy_input)
    target = torch.randint(4, size=(1, 2, 2))
    model = PixelCrossEntropyLoss()
    out = model(dummy_input, target)
    print(out)
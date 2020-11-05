import torch

labels = torch.LongTensor([1, 3])
targets = torch.zeros(2, 5)
out = targets.scatter_(1, torch.unsqueeze(labels, dim=-1), 1)
print(torch.unsqueeze(labels, dim=-1))
print(out)
# 注意dim=1，即逐样本的进行列填充
# 返回值为 tensor([[0, 1, 0, 0, 0],
#        [0, 0, 0, 1, 0]])
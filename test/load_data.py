import torch
import numpy as np
import os

# path = '../'
# filenames = os.listdir(path)
# for filename in filenames:
#     if 'set' in filename:
#         data = np.load(path+filename)
#         data = torch.Tensor(data)
#         torch.save(data, filename.split(".")[0]+".pt")


path = './'
filenames = os.listdir(path)
for filename in filenames:
    if 'set' in filename:
        data = torch.load(path+filename)
        print(data.size())
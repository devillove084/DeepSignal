import numpy as np
import torch
from GooleNet import GoogleNet

data = np.random.rand(256, 256)

net = GoogleNet()

#input = torch.Tensor(data)
input = torch.randn(64, 3, 7, 7)
out = net(input)

print(out)

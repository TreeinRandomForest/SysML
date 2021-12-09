import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

X = torch.from_numpy(np.array(train_img[0:3])).unsqueeze(1).float()

l1 = nn.Conv2d(1, 32, kernel_size=5)

act = nn.ReLU()

act(l1(x))
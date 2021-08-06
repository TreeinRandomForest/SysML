import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np

#Step 1: Construct a dataset
'''
torch.utils.data
'''
ds_train = MNIST(root='./', train=True, transform=ToTensor())
ds_test = MNIST(root='./', train=False, transform=ToTensor())

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyDataset(Dataset):
    def __init__(self, N):
        self.array = []

        #wouldn't do this
        self.construct_dataset(N)

        #read from the trace files
        #

    def construct_dataset(self, N):
        for i in range(N):
            inp = np.ones((2,2))*i
            out = i

            self.array.append((inp, out))

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        return self.array[i]

#Neural network
net = nn.Sequential(nn.Linear(3, 15),
                    nn.ReLU(),
                    nn.Linear(15, 30),
                    nn.ReLU(),
                    nn.Linear(30, 1))

#net(inp) -> out
#net.parameters() -> weights, biases
'''
layers: 

'''

net = nn.Sequential(nn.Conv2d(1, 32, 3),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(61952, 10),
                    #nn.Softmax(dim=1)
                    )

#Dataloader
#Define a neural net - class

criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(net.parameters(), lr=5e-2) #gradient descent

N_epochs = 20
net = net.to(device)

for i in range(N_epochs):

    total_loss = 0
    for idx, (X, y) in enumerate(ds_train):
        X = X.unsqueeze(0).to(device)
        y = torch.tensor(y).unsqueeze(0).to(device)

        X = 2*X - 1

        pred = net(X) #X = pixels

        
        loss = criterion(pred, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'epoch={i} loss={total_loss}')
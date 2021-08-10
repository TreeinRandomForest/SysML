import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, PILToTensor
import numpy as np

#Step 1: Construct a dataset
'''
torch.utils.data
'''
ds_train = MNIST(root='./', train=True, transform=PILToTensor())
ds_test = MNIST(root='./', train=False, transform=PILToTensor())

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
                    nn.Linear(30, 3))

#net(inp) -> out
#net.parameters() -> weights, biases
'''
layers: 

'''

net = nn.Sequential(nn.Conv2d(1, 16, 3),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(30976, 10),
                    #nn.Softmax(dim=1)
                    )

class Net(nn.Module):
    def __init__(self):
        super(Net).__init__()

        self.layer1 = nn.Conv2d(1, 32, 3)
        self.layer2 = nn.Conv2d(32, 64, 3)
        self.layer3 = nn.Conv2d(64, 128, 3)

        self.linear = nn.Linear(61952, 10)

        self.activation = nn.ReLU()


    def forward(self, x):
        #logic goes here

        return x

#Dataloader
#Define a neural net - class

criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(net.parameters(), lr=1e-3) #gradient descent

N_epochs = 20
net = net.to(device)

BATCH_SIZE = 64

dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE)

for i in range(N_epochs): #number of loops over full dataset

    total_loss = 0
    for idx, (X, y) in enumerate(dl_train): #picking 16 images at a time
        X = X.to(device)
        y = torch.tensor(y).to(device)

        X = (X - 128) / 255.

        pred = net(X) #predictions on 16 images
        loss = criterion(pred, y) #loss on 16 imagets

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f'Epoch = {i} Batch id = {idx} Loss = {loss}')

    print("-------------------Total Loss = {}".format(total_loss))

    '''
    for idx, (X, y) in enumerate(ds_train):
        X = X.unsqueeze(0).to(device)
        y = torch.tensor(y).unsqueeze(0).to(device)

        X = X/255.

        pred = net(X) #X = pixels

        
        loss = criterion(pred, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    '''
    #print(f'epoch={i} loss={total_loss}')
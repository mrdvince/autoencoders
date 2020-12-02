import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets
import torchvision.transfroms as transforms

# convert to tensor
transform = transforms.ToTensor()

train_data = datasets.MNIST('data', train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST('data', train=False,
                           download=True, transform=transform)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, num_workers=num_workers)

# Linear autoencoder


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init()__

        # encoder
        # linear layer
        self.fc1 = nn.Linear(28*28, encoding_dim)
        # decoder
        self.fc2 = nn.Linear(encoding_dim, 28*28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


# initialize model
encoding_dim = 32
model = Autoencoder(encoding_dim)
print(model)

# Convolutional Autoencoder
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # encoder layers
        # 1 --> 16
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 16 --> 4
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # maxpool
        self.pool = nn.MaxPool2d(2, 2)

        # # decoder layers
        # self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        # Alternarive to using transpose convolutions
        self.conv3 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # # encode
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # # decode
        # x = F.relu(self.t_conv1(x))
        # x = torch.sigmoid(self.t_conv2(x))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # decoder
        # using upsampling
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3(x))
        # Upsample again
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv4(x))
        return x


model = ConvAutoencoder()
print(model)

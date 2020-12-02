# Convolutional Autoencoder
import torch.nn.functional as F
import torch.nn as nn
import torch

# Linear autoencoder


class LinearAutoencoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super(LinearAutoencoder, self).__init__()

        # encoder
        # linear layer
        self.fc1 = nn.Linear(28*28, encoding_dim)
        # decoder
        self.fc2 = nn.Linear(encoding_dim, 28*28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Convolution autoencoder


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
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3(x))
        # Upsample again
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv4(x))
        return x

# define the NN architecture


class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will
        # increase the spatial dims by 2
        # kernel_size=3 to get to a 7x7 image output
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)
        # two more transpose layers with a kernel of 2
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # one, final, normal conv layer to decrease the depth
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation

        # decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))

        return x

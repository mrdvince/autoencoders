import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

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
        super(Autoencoder, self).__init__()

        # encoder
        # linear layer
        self.fc1 = nn.Linear(28*28, encoding_dim)
        # decoder
        self.fc2 = nn.Linear(encoding_dim, 28*28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# initialize model
encoding_dim = 32
model = Autoencoder(encoding_dim)
print(model)

# training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(1, epochs+1):
    # monitor loss
    train_loss = 0.0

    for data in train_loader:
        images, _ = data
        # flatten
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
    # print some training stats
    train_loss = train_loss/len(train_loader)
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.4f}')

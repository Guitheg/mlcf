import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration


class ConvNet(nn.Module):
    def __init__(self, nb_channel):
        super(ConvNet, self).__init__()
        k1,k2 = 3, 2
        self.conv1 = nn.Conv1d(nb_channel, nb_channel*2, k1)
        # self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(nb_channel*2, nb_channel*4, k2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # -> n, 3, 32, 32
        print(x.shape)
        print(x)
        x = F.relu(self.conv1(x))  
        print(x.shape)
        x = F.relu(self.conv2(x)) 
        print(x.shape)
        x = x.view(-1, 16 * 5 * 5)            
        print(x.shape)
        x = F.relu(self.fc1(x))              
        print(x.shape)
        x = F.relu(self.fc2(x))              
        print(x.shape)
        x = self.fc3(x)                       
        return x

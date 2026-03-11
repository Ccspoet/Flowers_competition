import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineFlowerModel(nn.Module):
    """
    A simple 1-layer CNN used as a baseline for the 
    Butterfly/Flower classification competition.
    """
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 112 * 112, 5) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x

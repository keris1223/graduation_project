# model.py
import torch.nn as nn
import torch.nn.functional as F
import torch
"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 10, 3)
        self.fc = nn.Linear(26 * 26 * 10, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""
class BigCNN(nn.Module):
    def __init__(self):
        super(BigCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)   # 1→64
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1) # 64→128
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1) # 128→256
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# models/cnn.py

import torch
import torch.nn as nn
from config import INPUT_SHAPE, NUM_CLASSES

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 48x48 -> 24x24 -> 12x12
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def build_cnn():
    """
    Build a simple CNN model.
    """
    return CNN()
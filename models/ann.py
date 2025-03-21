# models/ann.py

import torch
import torch.nn as nn
from config import INPUT_SHAPE, NUM_CLASSES

# class ANN(nn.Module):
#     def __init__(self):
#         super(ANN, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(INPUT_SHAPE[1] * INPUT_SHAPE[2], 128)
#         self.dropout1 = nn.Dropout(0.5)  # Dropout with 50% probability
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout(0.5)  # Dropout with 50% probability
#         self.fc3 = nn.Linear(64, NUM_CLASSES)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)  # Apply dropout
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)  # Apply dropout
#         x = self.fc3(x)
#         return x
# self.fc1 = nn.Linear(48 * 48, 1024)  # Increase neurons
# self.fc2 = nn.Linear(1024, 512)
# self.fc3 = nn.Linear(512, 256)
# self.fc4 = nn.Linear(256, 128)
# self.fc5 = nn.Linear(128, 64)
# self.fc6 = nn.Linear(64, 32)
# self.fc7 = nn.Linear(32, 7)  # Output layer

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(48 * 48, 1024)  # Increased neurons
        self.bn1 = nn.BatchNorm1d(1024)      # Batch normalization
        self.dropout1 = nn.Dropout(0.5)     # Dropout
        # Second hidden layer
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        
        # 3 hidden layer
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        
        # 4 hidden layer
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        
        # 5 hidden layer
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.5)
        
        # Fifth hidden layer
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(0.5)
        
        # Output layer
        self.fc7 = nn.Linear(32, 7)  # 7 classes

    def forward(self, x):
        x = self.flatten(x)
        
        # Layer 1
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Layer 4
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        # Layer 5
        x = torch.relu(self.fc5(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        
        x = torch.relu(self.fc6(x))
        x = self.bn6(x)
        x = self.dropout6(x)
        
        # Output layer
        x = self.fc7(x)
        return x


def build_ann():
    """
    Build a simple ANN model.
    """
    return ANN()


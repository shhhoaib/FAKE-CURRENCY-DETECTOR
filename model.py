"""Simple CNN Model for Currency Detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CurrencyCNN(nn.Module):
    """Simple CNN for fake/real currency classification."""
    
    def __init__(self, num_classes=2):
        super(CurrencyCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 poolings: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1: 224x224 -> 112x112
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2: 112x112 -> 56x56
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3: 56x56 -> 28x28
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_model(num_classes=2):
    """Create and return the CNN model."""
    return CurrencyCNN(num_classes=num_classes)

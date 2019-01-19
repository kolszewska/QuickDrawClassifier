from torch import nn
import torch.nn.functional as f


class Convolutional(nn.Module):
    def __init__(self, num_classes, dropout):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(x)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


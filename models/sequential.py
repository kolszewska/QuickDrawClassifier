from torch import nn
import torch.nn.functional as f


class Sequential(nn.Module):
    def __init__(self, num_classes):
        super(Sequential, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

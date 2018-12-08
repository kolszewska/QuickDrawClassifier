import torch
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the training data
train_set = datasets.ImageFolder('out/train/')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Load the test data
test_set = datasets.ImageFolder('out/test/')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

image, label = next(iter(train_loader))
helper.imshow(image[0, :])

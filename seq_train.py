import torch
from torch import nn, optim
from torchvision import transforms, datasets

# Hyper parameters
from models import Sequential

num_epochs = 5
num_classes = 10
batch_size = 256
learning_rate = 0.01

# Obtaining mean value from the images
#
# train_set = datasets.ImageFolder(root='out/train/', transform=transforms.ToTensor())
# validation_set = datasets.ImageFolder(root='out/validation/', transform=transforms.ToTensor())
# image_means = torch.stack([t.mean(1).mean(1) for t, c in train_set]
#                          + [t.mean(1).mean(1) for t, c in validation_set])

# Define a transform to normalize the data
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.8255, 0.8255, 0.8255], [0.0478, 0.0478, 0.0478])])

# Load the training data
train_set = datasets.ImageFolder('out2/train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Load the test data
test_set = datasets.ImageFolder('out2/test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Load the validation data
validation_set = datasets.ImageFolder('out2/validation/', transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# Create model
model = Sequential(num_classes)

# Define the loss
criterion = nn.CrossEntropyLoss()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, test_losses = [], []
batch_number = 0
total_training_batches = int(len(train_set) / batch_size)
for e in range(num_epochs):
    running_loss = 0
    for images, labels in train_loader:
        if batch_number % 10 == 0:
            print('Batch number {}/{}...'.format(batch_number, total_training_batches))
        batch_number += 1

        # Flatten images
        images = images.view(images.shape[0], -1)

        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        # Forwards pass, then backward pass, then update weights
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    step_number = 0
    test_loss = 0
    accuracy = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        for images, labels in test_loader:
            # Flatten images
            images = images.view(images.shape[0], -1)

            probabilities = model(images)
            test_loss += criterion(probabilities, labels)

            # Get the class probabilities
            ps = torch.softmax(probabilities, dim=1)

            # Get top probabilities
            top_probability, top_class = ps.topk(1, dim=1)

            # Comparing one element in each row of top_class with
            # each of the labels, and return True/False
            equals = top_class == labels.view(*top_class.shape)

            # Number of correct predictions
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Set model to train mode
        model.train()

        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))

        print("Epoch: {}/{}.. ".format(e + 1, num_epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

        batch_number = 0

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from models import Convolutional

# Hyper parameters

num_epochs = 100
num_classes = 10
batch_size = 256
learning_rate = 0.01

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.8255, 0.8255, 0.8255], [0.0478, 0.0478, 0.0478])])

# Load the training data
train_set = datasets.ImageFolder(root='out/train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Load the validation data
validation_set = datasets.ImageFolder(root='out/validation/', transform=transform)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

# Load the test data
test_set = datasets.ImageFolder(root='out/test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Create model
model = Convolutional(num_classes)
print(model)

# Define the loss
criterion = nn.CrossEntropyLoss()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, validation_losses = [], []
batch_number = 0
total_training_batches = int(len(train_set) / batch_size)
for e in range(num_epochs):
    running_loss = 0
    for images, labels in train_loader:
        if batch_number % 10 == 0:
            print('Batch number {}/{}...'.format(batch_number, total_training_batches))
        batch_number += 1

        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()

        # Forwards pass, then backward pass, then update weights
        probabilities = model(images)
        loss = criterion(probabilities, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    validation_loss = 0
    min_validation_loss = 1000
    accuracy = 0

    # Turn off gradients for testing
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        for images, labels in validation_loader:
            probabilities = model(images)
            validation_loss += criterion(probabilities, labels)

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
        validation_losses.append(validation_loss / len(validation_loader))

        # Get minimum validation loss
        min_validation_loss = min(validation_losses)

        # Counting the losses
        training_loss = running_loss / len(validation_loader)
        validation_loss = validation_loss / len(validation_loader)

        batch_number = 0

        print("Epoch: {}/{}.. ".format(e + 1, num_epochs),
              "Training Loss: {:.3f}.. ".format(training_loss),
              "Validation Loss: {:.3f}.. ".format(validation_loss),
              "Validation Accuracy: {:.3f}%".format((accuracy / len(validation_loader)) * 100))

        # Save model if validation loss have decreased
        if validation_loss <= min_validation_loss:
            print("Validation has decreased, saving model")
            torch.save(model.state_dict(), 'model.pt')
            min_validation_loss = validation_loss

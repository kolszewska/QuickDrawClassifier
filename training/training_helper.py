import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules import loss
from tensorboardX import SummaryWriter

writer = SummaryWriter()
logging.basicConfig(filename='convolutional_training.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(train_loader: DataLoader, validation_loader: DataLoader, test_loader: DataLoader, num_epochs: int,
          total_training_batches: int, model: Module, criterion: loss, optimizer: optim):
    """Train network."""
    batch_number = 0
    step_number = 0
    for epoch in range(num_epochs):
        running_loss = 0
        for images, labels in train_loader:
            if batch_number % 10 == 0:
                logging.info('Batch number {}/{}...'.format(batch_number, total_training_batches))
            batch_number += 1
            step_number += 1

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forwards pass, then backward pass, then update weights
            probabilities = model(images)
            model_loss = criterion(probabilities, labels)
            model_loss.backward()
            optimizer.step()

            running_loss += model_loss.item()

        validate(model, validation_loader, criterion, step_number)

        # Set model to train mode
        model.train()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('data/train_loss', train_loss, step_number)

        logging.info("Epoch: {}/{}.. ".format(epoch + 1, num_epochs))
        logging.info("Training Loss: {:.3f}.. ".format(train_loss))

        batch_number = 0

    test(model, test_loader)
    writer.close()


def validate(model: Module, validation_loader: DataLoader, criterion: loss, step_number: int):
    """Validate network."""
    accuracy = 0
    validation_loss = 0
    validation_losses = []
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        for images, labels in validation_loader:
            probabilities = model(images)
            validation_loss += criterion(probabilities, labels)

            # Get the class probabilities
            class_probabilities = torch.softmax(probabilities, dim=1)

            # Get top probabilities
            top_probability, top_class = class_probabilities.topk(1, dim=1)

            # Comparing one element in each row of top_class with
            # each of the labels, and return True/False
            equals = top_class == labels.view(*top_class.shape)

            # Number of correct predictions
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    # Set model to train mode
    model.train()

    validation_loss = validation_loss / len(validation_loader)
    validation_accuracy = (accuracy / len(validation_loader)) * 100

    writer.add_scalar('data/validation_loss', validation_loss, step_number)
    writer.add_scalar('data/validation_accuracy', validation_accuracy, step_number)

    validation_losses.append(validation_loss)

    logging.info("Validation Loss: {:.3f}.. ".format(validation_loss))
    logging.info("Validation Accuracy: {:.3f}%".format(validation_accuracy))

    # Get minimum validation loss
    min_validation_loss = min(validation_losses)

    # Save model if validation loss have decreased
    if validation_loss <= min_validation_loss:
        logging.info(
            "Validation has decreased {:.5f} -> {:.5f}, saving model".format(validation_loss,
                                                                             min_validation_loss))
        torch.save(model.state_dict(), 'model.pt'.format(min_validation_loss))


def test(model: Module, test_loader: DataLoader):
    """Test network."""
    # Prepare model for evaluation
    model.eval()

    classification_result_counter = 0
    test_accuracy = 0

    for images, labels in test_loader:

        # Flatten images
        probabilities = model(images)

        # Get the class probabilities
        ps = torch.softmax(probabilities, dim=1)

        # Get top probabilities
        top_probability, top_class = ps.topk(1, dim=1)

        # Comparing one element in each row of top_class with
        # each of the labels, and return True/False
        equals = top_class == labels.view(*top_class.shape)

        # Number of correct predictions
        test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Plot the image and probabilities
        if classification_result_counter <= 9:
            view_classification_result(images[0], ps[0])
            classification_result_counter += 1

    test_accuracy = (test_accuracy / len(test_loader)) * 100
    writer.add_scalar('data/test_accuracy', test_accuracy)
    logging.info("Test Accuracy: {:.3f}%".format(test_accuracy))


def view_classification_result(image, probabilities):
    """Viewing an image and it's predicted classes."""
    probabilities = probabilities.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['Bear', 'Bee', 'Camel', 'Cat', 'Cow', 'Crab', 'Crocodile', 'Dog', 'Dolphin', 'Duck'])
    ax2.set_title('Class probabilities')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

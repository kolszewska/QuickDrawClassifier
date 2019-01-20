import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules import loss
from tensorboardX import SummaryWriter

from models.convolutional import Convolutional

writer = SummaryWriter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(train_loader: DataLoader, validation_loader: DataLoader, num_epochs: int,
          total_training_batches: int, model: Module, criterion: loss, optimizer: optim, batch_size: int,
          learning_rate: float):
    """Train network."""

    writer.add_text('Experiment summary', 'Batch size: {}, Learning rate {}'.format(batch_size, learning_rate))

    batch_number = 0
    step_number = 0
    previous_running_loss = 0

    for epoch in range(num_epochs):
        train_running_loss = 0
        train_accuracy = 0
        # scheduler.step()  TODO
        for images, labels in train_loader:
            if batch_number % 10 == 0:
                logging.info('Batch number {}/{}...'.format(batch_number, total_training_batches))

            batch_number += 1
            step_number += 1

            # Pass this computations to selected device
            images = images.cuda()
            labels = labels.cuda()

            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            # Forwards pass, then backward pass, then update weights
            probabilities = model.forward(images)
            model_loss = criterion(probabilities, labels)
            model_loss.backward()
            optimizer.step()

            # Get the class probabilities
            ps = torch.nn.functional.softmax(probabilities, dim=1)

            # Get top probabilities
            top_probability, top_class = ps.topk(1, dim=1)

            # Comparing one element in each row of top_class with
            # each of the labels, and return True/False
            equals = top_class == labels.view(*top_class.shape)

            # Number of correct predictions
            train_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()
            train_running_loss += model_loss.item()
        else:
            validation_running_loss = 0
            validation_accuracy = 0
            # Turn off gradients for testing
            with torch.no_grad():
                # set model to evaluation mode
                model.eval()
                for images, labels in validation_loader:
                    # Pass this computations to selected device
                    images = images.cuda()
                    labels = labels.cuda()

                    probabilities = model.forward(images)
                    validation_running_loss += criterion(probabilities, labels)

                    # Get the class probabilities
                    ps = torch.nn.functional.softmax(probabilities, dim=1)

                    # Get top probabilities
                    top_probability, top_class = ps.topk(1, dim=1)

                    # Comparing one element in each row of top_class with
                    # each of the labels, and return True/False
                    equals = top_class == labels.view(*top_class.shape)

                    # Number of correct predictions
                    validation_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

            if validation_running_loss <= previous_running_loss:
                logging.info(
                    'Validation loss decreased {:.5f} -> {:.5f}. Saving model.'.format(previous_running_loss,
                                                                                       validation_running_loss))
                torch.save(model.state_dict(), 'model_{}.pt'.format(batch_size))

            previous_running_loss = validation_running_loss

            # Set model to train mode
            model.train()

            # Calculating accuracy
            validation_accuracy = (validation_accuracy / validation_loader.sampler.num_samples * 100)
            train_accuracy = (train_accuracy / train_loader.batch_sampler.sampler.num_samples * 100)

            # Saving losses and accuracy
            writer.add_scalar('data/train_loss', train_running_loss, epoch)
            writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
            writer.add_scalar('data/validation_loss', validation_running_loss, epoch)
            writer.add_scalar('data/validation_accuracy', validation_accuracy, epoch)

            logging.info("Epoch: {}/{}.. ".format(epoch + 1, num_epochs))
            logging.info("Training Loss: {:.3f}.. ".format(train_running_loss))
            logging.info("Training Accuracy: {:.3f}%".format(train_accuracy))
            logging.info("Validation Loss: {:.3f}.. ".format(validation_running_loss))
            logging.info("Validation Accuracy: {:.3f}%".format(validation_accuracy))

            batch_number = 0


def test(test_loader: DataLoader, model: Convolutional):
    """Test network."""
    model.eval()
    logging.info("Testing the model...")

    running_accuracy = 0
    step_number = 0

    for images, labels in test_loader:

        if step_number % 10 == 0:
            logging.info('Testing....')

        step_number += 1

        probabilities = model(images)

        # Get the class probabilities
        ps = torch.softmax(probabilities, dim=1)

        # Get top probabilities
        top_probability, top_class = ps.topk(1, dim=1)

        # Comparing one element in each row of top_class with
        # each of the labels, and return True/False
        equals = top_class == labels.view(*top_class.shape)

        # Number of correct predictions
        running_accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

    test_accuracy = (running_accuracy / test_loader.sampler.num_samples * 100)
    writer.add_scalar('data/test_accuracy', test_accuracy)
    logging.info("Test Accuracy: {:.3f}%".format(test_accuracy))

    writer.close()


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

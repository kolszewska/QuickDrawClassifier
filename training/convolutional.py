from torch import optim
from torch.nn.modules import loss

from models.convolutional import Convolutional
from training.data_helper import get_transforms_for_convolutional, load_data
from training.training_helper import train, test

NUM_EPOCHS = 50
NUM_CLASSES = 10

BATCH_SIZE = 512
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DROPOUT = 0.8

train_loader, validation_loader, test_loader, total_training_batches = load_data('out/train', 'out/validation',
                                                                                 'out/test', BATCH_SIZE,
                                                                                 get_transforms_for_convolutional())
model = Convolutional(NUM_CLASSES, DROPOUT)
criterion = loss.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

train(train_loader, validation_loader, NUM_EPOCHS, total_training_batches, model, criterion, optimizer)
test(test_loader)

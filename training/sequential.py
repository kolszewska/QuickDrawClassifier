from torch import optim
from torch.nn.modules import loss

from models.sequential import Sequential
from training.data_helper import get_transforms_for_sequential, load_data
from training.training_helper import train

NUM_EPOCHS = 5
NUM_CLASSES = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.01

train_loader, validation_loader, test_loader, total_training_batches = load_data('out/train', 'out/validation',
                                                                                 'out/test', BATCH_SIZE,
                                                                                 get_transforms_for_sequential())
model = Sequential(NUM_CLASSES)
optimizer = loss.CrossEntropyLoss()
criterion = optim.SGD(model.parameters(), lr=LEARNING_RATE)

train(train_loader, validation_loader, test_loader, NUM_EPOCHS, total_training_batches, model, criterion, optimizer)

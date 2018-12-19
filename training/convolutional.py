from torch import optim
from torch.nn.modules import loss

from models.convolutional import Convolutional
from training.data_helper import get_transforms_for_convolutional, load_data
from training.training_helper import train

NUM_EPOCHS = 50
NUM_CLASSES = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.01

train_loader, validation_loader, test_loader, total_training_batches = load_data('out2/train', 'out2/validation',
                                                                                 'out2/test', BATCH_SIZE,
                                                                                 get_transforms_for_convolutional())
model = Convolutional(NUM_CLASSES)
criterion = loss.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

train(train_loader, validation_loader, test_loader, NUM_EPOCHS, total_training_batches, model, criterion, optimizer)

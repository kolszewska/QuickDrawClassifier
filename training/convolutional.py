import sys

import torch
from torch import optim
from torch.nn.modules import loss

from models.convolutional import Convolutional
from training.data_helper import get_transforms_for_convolutional, load_data
from training.training_helper import train, test

torch.cuda.manual_seed(13)  # Making sure that we have the same initial weights in every experiment

num_epochs = 50
num_classes = 10

batch_size = int(sys.argv[1])
learning_rate = 0.0001
dropout = 0.5

train_loader, validation_loader, test_loader, total_training_batches = load_data('out/train', 'out/validation',
                                                                                 'out/test', batch_size,
                                                                                 get_transforms_for_convolutional())
model = Convolutional(num_classes, dropout)
model = model.cuda()  # Move model to CUDA
criterion = loss.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(train_loader, validation_loader, num_epochs, total_training_batches, model, criterion, optimizer, batch_size,
      learning_rate)

loaded_model = Convolutional(num_classes, dropout)
loaded_model.load_state_dict(torch.load('model_{}.pt'.format(batch_size)))
test(test_loader, loaded_model)

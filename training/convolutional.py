import torch
from torch import optim
from torch.nn.modules import loss

from models.convolutional import Convolutional
from training.data_helper import get_transforms_for_convolutional, load_data
from training.training_helper import train, test

num_epochs = 50
num_classes = 10

batch_size = 32
learning_rate = 0.001
momentum = 0.9
dropout = 0.5

scheduler_step_size = 5
scheduler_gamma = 0.1

train_loader, validation_loader, test_loader, total_training_batches = load_data('out/train', 'out/validation',
                                                                                 'out/test', batch_size,
                                                                                 get_transforms_for_convolutional())
model = Convolutional(num_classes, dropout)
model = model.cuda()  # Move model to CUDA
criterion = loss.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

train(train_loader, validation_loader, num_epochs, total_training_batches, model, criterion, optimizer, scheduler)

loaded_model = Convolutional(num_classes, dropout)
loaded_model.load_state_dict(torch.load('model.pt'))

test(test_loader, loaded_model)

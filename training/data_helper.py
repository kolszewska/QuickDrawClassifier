import torch
from torchvision import transforms, datasets


def load_data(train_data_path: str, validation_data_path: str, test_data_path: str, batch_size: int,
              transform: transforms):
    """Load the data from the given folders and return its loaders."""
    train_set = datasets.ImageFolder(root=train_data_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validation_set = datasets.ImageFolder(validation_data_path, transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    total_training_batches = int(len(train_set) / batch_size)

    return train_loader, validation_loader, test_loader, total_training_batches


def get_mean_values(train_set: datasets.ImageFolder, validation_set: datasets.ImageFolder):
    """ Get mean values from the train and validation set.

    This value will be later used for normalization of the pictures.
    """
    return torch.stack([t.mean(1).mean(1) for t, c in train_set]
                       + [t.mean(1).mean(1) for t, c in validation_set])


def get_transforms_for_sequential():
    """Return transforms for data that will be used in Sequential network."""
    return transforms.Compose([transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),  # TODO: resize tensor from [28,28] to [784]
                               transforms.Normalize([0.8255, 0.8255, 0.8255], [0.0478, 0.0478, 0.0478])])


def get_transforms_for_convolutional():
    """Return transforms for data that will be used in Convolutional network."""
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.8255, 0.8255, 0.8255], [0.0478, 0.0478, 0.0478])])

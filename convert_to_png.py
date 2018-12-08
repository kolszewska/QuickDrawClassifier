import os

import numpy as np
from PIL import Image, ImageOps

npy_dir = './in/'
out_dir = './out/'

npy_files = [f for f in os.listdir(npy_dir) if os.path.isfile(os.path.join(npy_dir, f))]
categories = []

# Get categories from file names
for file in npy_files:
    category_split = file.split('_')[3].split('.')[0]
    category = category_split.title()
    categories.append(category)

# Create directories
for category in categories:
    os.makedirs(os.path.join(out_dir, 'train', category))
    os.makedirs(os.path.join(out_dir, 'test', category))
    os.makedirs(os.path.join(out_dir, 'validation', category))

# Split pictures to train, test and validation folders
category_index = 0
for file in npy_files:
    images = np.load(os.path.join(npy_dir, file))
    train_images = range(0, 800, 1)
    test_images = range(800, 1000, 1)
    validation_images = range(1000, 1150, 1)

    for i in train_images:
        file_name = '{}.png'.format(i + 1)
        file_path = os.path.join(out_dir, 'train', categories[category_index], file_name)
        img = images[i].reshape(28, 28)
        f_img = Image.fromarray(img)
        im = ImageOps.invert(f_img)
        im.save(file_path, 'png')

    for i in test_images:
        file_name = '{}.png'.format(i + 1)
        file_path = os.path.join(out_dir, 'test', categories[category_index], file_name)
        img = images[i].reshape(28, 28)
        f_img = Image.fromarray(img)
        im = ImageOps.invert(f_img)
        im.save(file_path, 'png')

    for i in validation_images:
        file_name = '{}.png'.format(i + 1)
        file_path = os.path.join(out_dir, 'validation', categories[category_index], file_name)
        img = images[i].reshape(28, 28)
        f_img = Image.fromarray(img)
        im = ImageOps.invert(f_img)
        im.save(file_path, 'png')

    category_index += 1

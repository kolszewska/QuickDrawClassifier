import os

import numpy as np
from PIL import Image, ImageOps

npy_dir = './in/'
out_dir = './out/'
npy_files = [f for f in os.listdir(npy_dir) if os.path.isfile(os.path.join(npy_dir, f))]


def convert_images(number_of_images: range, images: any, folder_name: str, index: int):
    for i in number_of_images:
        file_name = '{}.png'.format(i + 1)
        file_path = os.path.join(out_dir, folder_name, categories[index], file_name)
        img = images[i].reshape(28, 28)
        f_img = Image.fromarray(img)
        im = ImageOps.invert(f_img)
        im.save(file_path, 'png')


# Get categories from file names
categories = []
for file in npy_files:
    category_split = file.split('_')[3].split('.')[0]
    category = category_split.title()
    categories.append(category)

# Create directories
for category in categories:
    os.makedirs(os.path.join(out_dir, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'validation', category), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'test', category), exist_ok=True)

# Split pictures to train, test and validation folders
category_index = 0
for file in npy_files:
    images_from_npy = np.load(os.path.join(npy_dir, file))
    train_images = range(0, 20000, 1)
    validation_images = range(20000, 24000, 1)
    test_images = range(24000, 25000, 1)

    convert_images(train_images, images_from_npy, 'train', category_index)
    convert_images(validation_images, images_from_npy, 'validation', category_index)
    convert_images(test_images, images_from_npy, 'test', category_index)

    category_index += 1

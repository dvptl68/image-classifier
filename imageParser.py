import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Retrieve dataset
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip', extract=True)

# Set dataset paths
train_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train')
validation_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Useful variables
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Training and validation data generator
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

sample_training_images, _ = next(train_data_gen)

# Image plotter
# def plotImages(images_arr):
#   _, axes = plt.subplots(1, 5, figsize=(20,20))
#   axes = axes.flatten()
#   for img, ax in zip( images_arr, axes):
#     ax.imshow(img)
#     ax.axis('off')
#   plt.tight_layout()
#   plt.show()

# plotImages(sample_training_images[:5])

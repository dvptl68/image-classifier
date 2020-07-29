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
train_cats_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'cats')
train_dogs_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'dogs')
validation_cats_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'cats')
validation_dogs_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'dogs')

# Useful variables
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Training and validation data generator
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

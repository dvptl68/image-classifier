import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Retrieve dataset
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip', extract=True)

# Delete created zip file
os.remove(path_to_zip)

# Set dataset paths
train_cats_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'cats')
train_dogs_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'dogs')
validation_cats_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'cats')
validation_dogs_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered', 'train', 'dogs')
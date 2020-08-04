from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os

# Dataset paths
train_dir = os.path.join('datasets', 'cats_and_dogs_filtered', 'train')
validation_dir = os.path.join('datasets', 'cats_and_dogs_filtered', 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Useful variables
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

input_shape = (3, IMG_WIDTH, IMG_HEIGHT) if K.image_data_format() == 'channels_first' else (IMG_WIDTH, IMG_HEIGHT, 3)

# Model information
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(loss ='binary_crossentropy', optimizer ='rmsprop', metrics =['accuracy'])

# Create image data generators
train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1. / 255)

# Generate from dataset
train_generator = train_datagen.flow_from_directory(train_dir, target_size =(IMG_WIDTH, IMG_HEIGHT), batch_size = batch_size, class_mode ='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size =(IMG_WIDTH, IMG_HEIGHT), batch_size = batch_size, class_mode ='binary')
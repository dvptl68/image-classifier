from keras.datasets import mnist
import tensorflow as tensor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
import tensorflow as tf
from keras._tf_keras.keras import models, layers
from typing import Tuple

def create_cnn_2d(input_shape: Tuple[int, int], n_filters: int, n_layers: int):
    model = models.Sequential()
    model.add(layers.Conv2D(n_filters, (3, 3), activation="relu", input_shape=(input_shape[0], input_shape[1], 1)))
    for layer in range(n_layers - 1):
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(n_filters, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(2))
    return model.summary()

def create_cnn_1d(input_shape: int, n_filters: int, n_layers: int):
    model = models.Sequential()
    model.add(layers.Conv1D(n_filters, 3, activation="relu", input_shape=(input_shape, 1)))
    return model.summary()

if __name__ == "__main__":
    print(create_cnn_2d((100, 200), 32, 4))
    # print(create_cnn_1d(100, 32, 4))

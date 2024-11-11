import tensorflow as tf
from keras._tf_keras.keras import models, layers, losses
from typing import Tuple

def create_cnn_2d(input_shape: Tuple[int, int], n_filters: int, n_layers: int, pooling=True) -> models.Sequential:
    """
    creates a 2D cnn with:
     - n_layers convolutional layers separated by max pooling layers
     - n_filters filters in each convolutional layer
     - a 2d input shape defined in a tuple (x, y)
    """
    model = models.Sequential()
    model.add(layers.Input((input_shape[0], input_shape[1], 1)))
    model.add(layers.Conv2D(n_filters, (3, 3), activation="relu"))
    for layer in range(n_layers - 1):
        if pooling:
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(n_filters, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.compile(optimizer="adam",
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model

def create_cnn_1d(input_shape: int, n_filters: int, n_layers: int, pooling=True) -> models.Sequential:
    """
    creates a 1D cnn with:
     - n_layers convolutional layers separated by max pooling layers
     - n_filters filters in each convolutional layer
     - a 1d input shape defined as an int
    """
    model = models.Sequential()
    model.add(layers.Input((input_shape, 1)))
    model.add(layers.Conv1D(n_filters, 3, activation="relu"))
    for layer in range(n_layers - 1):
        if pooling:
            model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(n_filters, 3, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.compile(optimizer="adam",
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = create_cnn_1d(10, 32, 2)
    # model = create_cnn_2d((10, 10), 32, 2)
    print(model.summary())

from numpy import concatenate
import tensorflow as tf
from keras._tf_keras.keras import models, layers, losses
from typing import Tuple

def create_cnn_2d(input_shape: Tuple[int, int],
                  n_filters: int,
                  n_layers: int,
                  pooling: bool = True,
                  output_size: int = -1
                  ) -> models.Sequential:
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
    if output_size > 0:
        model.add(layers.Dense(output_size))
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

def create_cnn_1d(input_shape: int,
                  n_filters: int,
                  n_layers: int,
                  pooling: bool = True,
                  output_size: int = -1
                  ) -> models.Sequential:
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
    if output_size > 0:
        model.add(layers.Dense(output_size))
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

def stitch_and_terminate(model_list: list[models.Sequential],
                         ) -> models.Model:
    outputs = []
    for m in model_list:
        outputs.append(m.layers[-1].output)
    inputs = []
    for m in model_list:
        inputs.append(m.inputs)
    stitch = layers.Concatenate()(outputs)
    stitch = layers.Flatten()(stitch)
    stitch = layers.Dense(1)(stitch)
    model = models.Model(inputs=inputs, outputs=stitch)
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # model = create_cnn_1d(10, 32, 2)
    part0 = create_cnn_2d((10, 10), 32, 2)
    part1 = create_cnn_2d((10, 10), 32, 2)
    model = stitch_and_terminate([part0, part1])
    print(model.summary())

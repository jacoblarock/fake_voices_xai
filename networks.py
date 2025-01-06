from numpy import concatenate
import tensorflow as tf
from keras._tf_keras.keras import models, layers, losses, utils, Model
from tensorflow import convert_to_tensor
from typing import Tuple
import numpy as np
import pickle

def create_cnn_2d(input_shape: Tuple[int, int],
                  n_filters: int,
                  n_layers: int,
                  pooling: bool = True,
                  output_size: int = -1,
                  name: str | None = None
                  ) -> models.Sequential:
    """
    creates a 2D cnn with the specified arguments
    Arguments
    - input_shape: the input shape of the cnn in a tuple (x, y)
    - n_filters: filters in each convolutional layer
    - n_layers: convolutional layers separated by max pooling layers
    Keyword arguments:
    - pooling: bool whether there should be pooling layers
    - output_size: the output size of the model
      If the value is -1 (default) there will not be a final dense layer and the size of the previous output will be final
    """
    model = models.Sequential()
    if name != None:
        model.add(layers.Input((input_shape[0], input_shape[1], 1), name="input_" + name))
    else:
        model.add(layers.Input((input_shape[0], input_shape[1], 1)))
    model.add(layers.Conv2D(n_filters, (3, 3), activation="relu"))
    for layer in range(n_layers - 1):
        if pooling:
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(n_filters, (3, 3), activation="relu"))
    if output_size > 0:
        model.add(layers.Flatten())
        if name != None:
            model.add(layers.Dense(output_size, name="out_" + name))
        else:
            model.add(layers.Dense(output_size))
    else:
        if name != None:
            model.add(layers.Flatten(name="out_" + name))
        else:
            model.add(layers.Flatten())
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

def create_cnn_1d(input_shape: int,
                  n_filters: int,
                  n_layers: int,
                  pooling: bool = True,
                  output_size: int = -1,
                  name: str | None = None
                  ) -> models.Sequential:
    """
    creates a 1D cnn with the specified arguments
    Arguments
    - input_shape: the input shape of the cnn in an int
    - n_filters: filters in each convolutional layer
    - n_layers: convolutional layers separated by max pooling layers
    Keyword arguments:
    - pooling: bool whether there should be pooling layers
    - output_size: the output size of the model
      If the value is -1 (default) there will not be a final dense layer and the size of the previous output will be final
    """
    model = models.Sequential()
    if name != None:
        model.add(layers.Input((input_shape, 1), name="input_" + name))
    else:
        model.add(layers.Input((input_shape, 1)))
    model.add(layers.Conv1D(n_filters, 3, activation="relu"))
    for layer in range(n_layers - 1):
        if pooling:
            model.add(layers.MaxPooling1D(2))
        model.add(layers.Conv1D(n_filters, 3, activation="relu"))
    if output_size > 0:
        model.add(layers.Flatten())
        if name != None:
            model.add(layers.Dense(output_size, name="out_" + name))
        else:
            model.add(layers.Dense(output_size))
    else:
        if name != None:
            model.add(layers.Flatten(name="out_" + name))
        else:
            model.add(layers.Flatten())
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

def single_input(name: str | None = None) -> models.Sequential:
    model = models.Sequential()
    if name != None:
        model.add(layers.Input((1, 1), name="input_" + name))
    else:
        model.add(layers.Input((1, 1)))
    model.add(layers.Flatten())
    if name != None:
        model.add(layers.Dense(1, name="out_" + name))
    else:
        model.add(layers.Dense(1))
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

def multi_input(in_count: int,
                out_count: int,
                name: str | None = None
                ) -> models.Sequential:
    """
    Creates a model with multiple single-input layers that are concatted together into a specified
    number of outputs
    Arguments:
    - in_count: number of input layers to create
    - out_count: number of output layers to create
    Keyword arguments:
    - name: optional name for the input/output layers
    """
    model = models.Sequential()
    if name != None:
        model.add(layers.Input((in_count, 1), name="input_" + name))
    else:
        model.add(layers.Input((in_count, 1)))
    model.add(layers.Flatten())
    if name != None:
        model.add(layers.Dense(out_count, name="out_" + name))
    else:
        model.add(layers.Dense(out_count))
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    return model

def stitch_and_terminate(model_list: list[models.Sequential],
                         ) -> models.Model:
    """
    Stitch multiple models (created using the above functions) together to use multiple (non-concatted) features
    Arguments:
    - model_list: list of models to stitch together
    """
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

def decompose(model: models.Model) -> dict[str, models.Model]:
    names = []
    out = {}
    for layer in model.layers:
        if layer.name[0:5] == "input":
            names.append(layer.name[6:])
    for name in names:
        inputs = model.get_layer("input_" + name).output
        outputs = model.get_layer("out_" + name).output
        out[name] = Model(inputs=inputs, outputs=outputs)
    try:
        inputs = model.get_layer("dense").input
        outputs = model.get_layer("dense").output
        out["terminus"] = Model(inputs=inputs, outputs=outputs)
    except Exception as e:
        print(e)
    return out

if __name__ == "__main__":
    model: Model = pickle.load(open("./trained_models/ItW_multi_percep_until10000", "rb"))
    print(model.summary())
    model_renames = {"input_mel": "input_mel_spec",
                     "input_mfcc": "input_mfccs",
                     "input_local_shimmer": "input_local_shim",
                     "input_rap_shimmer": "input_rap_shim",
                     "input_ppq5_shimmer": "input_ppq5_shim",
                     "input_ppq55_shimmer": "input_ppq55_shim",
                     "out_mel": "out_mel_spec",
                     "out_mfcc": "out_mfccs",
                     "out_local_shimmer": "out_local_shim",
                     "out_rap_shimmer": "out_rap_shim",
                     "out_ppq5_shimmer": "out_ppq5_shim",
                     "out_ppq55_shimmer": "out_ppq55_shim",
                     "dense_15": "dense"
                     }
    for name in model_renames:
        model.get_layer(name).name = model_renames[name]
    print(model.summary())
    utils.plot_model(model, "./cache/model.png", show_layer_names=True)
    pickle.dump(model, open("./trained_models/ItW_multi_percep_until10000", "wb"))

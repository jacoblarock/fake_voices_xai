from numpy import concatenate
import tensorflow as tf
from keras._tf_keras.keras import models, layers, losses, utils, Model
from typing import Tuple
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

def single_input(name: str | None = None):
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
    print(model.summary())
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

if __name__ == "__main__":
    model: Model = pickle.load(open("./trained_models/ItW_multi_percep_until10000", "rb"))
    print(model.summary())
    model_renames = {"input_layer_1": "input_mel",
                     "input_layer": "input_hnrs",
                     "input_layer_2": "input_mfcc",
                     "input_layer_3": "input_f0_lens",
                     "input_layer_4": "input_onset_strength",
                     "input_layer_5": "input_intensity",
                     "input_layer_6": "input_pitch_flucs",
                     "input_layer_7": "input_local_jitter",
                     "input_layer_8": "input_rap_jitter",
                     "input_layer_9": "input_ppq5_jitter",
                     "input_layer_10": "input_ppq55_jitter",
                     "input_layer_11": "input_local_shimmer",
                     "input_layer_12": "input_rap_shimmer",
                     "input_layer_13": "input_ppq5_shimmer",
                     "input_layer_14": "input_ppq55_shimmer",
                     "dense_1": "out_mel",
                     "dense": "out_hnrs",
                     "dense_2": "out_mfcc",
                     "dense_3": "out_f0_lens",
                     "dense_4": "out_onset_strength",
                     "dense_5": "out_intensity",
                     "dense_6": "out_pitch_flucs",
                     "dense_7": "out_local_jitter",
                     "dense_8": "out_rap_jitter",
                     "dense_9": "out_ppq5_jitter",
                     "dense_10": "out_ppq55_jitter",
                     "dense_11": "out_local_shimmer",
                     "dense_12": "out_rap_shimmer",
                     "dense_13": "out_ppq5_shimmer",
                     "dense_14": "out_ppq55_shimmer"}
    for name in model_renames:
        model.get_layer(name).name = model_renames[name]
    print(model.summary())
    utils.plot_model(model, "./cache/model.png", show_layer_names=True)
    pickle.dump(model, open("./trained_models/ItW_multi_percep_until10000", "wb"))

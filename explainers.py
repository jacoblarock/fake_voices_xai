import networks
import classification
from keras._tf_keras.keras import models, layers, Model
import pandas as pd
import numpy as np
import lime.lime_tabular as lt
import tensorflow as tf

def gen_intermediate_train_data(model,
                                features: list[pd.DataFrame],
                                feature_names: list[str],
                                batch_size: int
                                ) -> pd.DataFrame:
    out = pd.DataFrame()
    dec_model = networks.decompose(model)
    for sub_model in dec_model:
        print(sub_model)
    # remove the terminus model
    if "terminus" in dec_model.keys():
        del dec_model["terminus"]
    if len(dec_model) != len(features):
        print(len(dec_model))
        print(len(features))
        raise ValueError("number of features does not correspond to the number of sub models")
    for i in range(len(features)):
        feature_name = feature_names[i]
        sub_model = dec_model[feature_name]
        feature = features[i].loc[:, "feature"]
        print("gen intermediate data for", feature_name)
        batches = classification.gen_batches(feature.index, batch_size)
        for batch in batches:
            input_data = feature.loc[batch].to_numpy()
            input_data = np.stack(input_data, axis=0)
            input_data = tf.convert_to_tensor(input_data)
            batch_out = sub_model.predict(input_data)
            print(batch_out)
    return pd.DataFrame()

def explain_single_feature(model_slice: Model,
                           train_feature: pd.DataFrame,
                           samples: pd.Series):
    x = train_feature.loc[:, "feature"].to_numpy()
    x = np.stack(x, axis=0)
    explainer = lt.LimeTabularExplainer(x)
    explanations = []


def explain(model: Model,
            features: list[pd.DataFrame],
            feature_cols: list[str]
            ):
    prediction = classification.classify(model, features, feature_cols)

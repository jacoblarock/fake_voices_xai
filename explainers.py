import networks
import classification
from keras._tf_keras.keras import models, layers, Model
import pandas as pd
import numpy as np
import lime.lime_tabular as lt
import tensorflow as tf
from random import uniform as rand

def gen_intermediate_train_data(model,
                                features: list[pd.DataFrame],
                                feature_names: list[str],
                                batch_size: int
                                ) -> dict[str, pd.Series]:
    out = {}
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
        out[feature_name] = pd.Series()
        print("gen intermediate data for", feature_name)
        batches = classification.gen_batches(feature.index, batch_size)
        for batch in batches:
            input_data = feature.loc[batch].to_numpy()
            input_data = np.stack(input_data, axis=0)
            input_data = tf.convert_to_tensor(input_data)
            batch_out = sub_model.predict(input_data)
            batch_out = pd.Series(batch_out.tolist())
            out[feature_name] = pd.concat((out[feature_name], batch_out)).reset_index(drop=True)
            print(out[feature_name])
    return out

def inter_data_concat(inter_train_data: dict[str, pd.Series],
                             ) -> np.ndarray:
    out = pd.Series()
    for feature in inter_train_data:
        out = classification.additive_merge(out, inter_train_data[feature])
    out = out.to_numpy()
    print("start merge", classification.datetime.now())
    return np.array([np.concatenate(row) for row in out])

def explain_single_feature(model_slice: Model,
                           train_feature: pd.DataFrame,
                           samples: pd.Series):
    """
    Not fully implemented
    """
    x = train_feature.loc[:, "feature"].to_numpy()
    x = np.stack(x, axis=0)
    explainer = lt.LimeTabularExplainer(x)
    explanations = []


def explain(model: Model,
            features: list[pd.DataFrame],
            feature_cols: list[str],
            training_data: np.ndarray
            ) -> lt.explanation.Explanation:
    explainer = lt.LimeTabularExplainer(training_data=training_data,
                                        mode="classification")
    terminus = networks.decompose(model)["terminus"]
    out = explainer.explain_instance(data_row=np.array([rand(-100, 100) for x in range(218)]),
                                     predict_fn=terminus.predict)
    return out

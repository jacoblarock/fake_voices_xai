from typing import Tuple, Union
import numpy as np
import pandas as pd
import os
import pickle
import feature_extraction
import networks

def get_labels(path: str,
               name_col: str,
               label_col: str,
               label_0_val: str,
               label_1_val: str
               ) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.rename(columns={label_col: "label"})
    data = data.rename(columns={name_col: "name"})
    data["label"] = data["label"].replace(label_0_val, 0)
    data["label"] = data["label"].replace(label_1_val, 1)
    data = data.loc[:, ["name", "label"]]
    return data

def match_labels(labels: pd.DataFrame,
                 extracted_features: pd.DataFrame,
                 name: str,
                 cache: bool = True,
                 use_cached: bool = True
                 ) -> pd.DataFrame:
    if cache or use_cached:
        feature_extraction.check_cache()
    cache_path = "./cache/matched_labels" + name
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as file:
            extracted_features = pickle.load(file)
    else:
        extracted_features["label"] = None
        for i in labels.index:
            name = labels.loc[i, "name"]
            label = labels.loc[i, "label"]
            to_label = extracted_features[0] == name
            extracted_features.loc[to_label, "label"] = label
        if cache:
            with open(cache_path, "wb") as file:
                pickle.dump(extracted_features, file)
    return extracted_features


def train(matched_labels: pd.DataFrame,
             model: networks.models.Sequential,
             epochs: int
             ):
    x = np.array(list(matched_labels[2]))
    y = np.array(list(matched_labels["label"]))
    history = model.fit(x=x, y=y, epochs=epochs)
    return history

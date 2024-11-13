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
    """
    Returns labels from a csv file in a standardized dataframe. this is necessary for efficient label matching.
    Arguments:
     - path: the path of the label file
     - label_col: the name of the column containing labels in the file
     - label_0_val: value in the file of the result 0
     - label_1_val: value in the file of the result 1
    """
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
    """
    Modifies the extracted_features dataframe to include labels from a labels dataframe.
    This dataframe is also returned.
    Arguments:
     - labels: previously extracted label dataframe
     - name: name of the matched label set for caching purposes
    Keyword arguments:
     - cache: if True, data will be saved to the cache
     - use_cached: if True, previously cached data will be used, if it exists
    WARNING: The input extracted_features dataframe WILL be modified due to memory reasons
    """
    if cache or use_cached:
        feature_extraction.check_cache()
    cache_path = "./cache/matched_labels/" + name
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

# meant for internal use
# expand a one dimensional array into two dimensions
def morph(arr: np.ndarray, vsize: int) -> np.ndarray:
    if len(arr.shape) != 1:
        raise Exception("Vertical size of array to morph must be 1")
    out = np.ndarray((vsize, arr.shape[0]))
    for i in range(vsize):
        out[i] = arr.copy()
    return out

# TODO: Support for merging features with and without the sliding window
def merge(matched_labels: pd.DataFrame,
          feature: pd.DataFrame
          ) -> pd.DataFrame:
    """
    Modifies an input dataframe of matched labels to include a further feature.
    Arguments:
     - matched_labels: previously generated dataframe of matched labels and features(s)
     - feature: new feature to add to matched_labels
    WARNING: The input matched_labels dataframe WILL be modified due to memory reasons
    """
    if type(matched_labels[2][0]) == np.ndarray and type(feature[2][0]) == np.ndarray:
        matched_labels = matched_labels.sort_values(by=[0, 1])
        feature = feature.sort_values(by=[0, 1])
        # morph sizes if number of dimensions is different
        vsize_matched_labels = 1
        if len(matched_labels[2][0].shape) > 1:
            vsize_matched_labels = matched_labels[2][0].shape[1]
        vsize_feature = 1
        if len(feature[2][0].shape) > 1:
            vsize_feature = feature[2][0].shape[1]
        if vsize_matched_labels == 1 and vsize_feature != 1:
            matched_labels[2] = matched_labels[2].apply(morph, args=(vsize_feature,))
        if vsize_matched_labels != 1 and vsize_feature == 1:
            feature[2] = feature[2].apply(morph, args=(vsize_matched_labels,))
        # perform join
        matched_labels = matched_labels.join(feature.set_index([0]), on=[0], how="cross", rsuffix=".temp")
        # concat feature in 2 and feature in temp
        matched_labels["2"] = matched_labels[["2", "2.temp"]].apply(lambda row: np.concatenate((row["2"], row["2.temp"])), axis=1)
        # for i in matched_labels.index:
        #     a = matched_labels.loc[i, "2"]
        #     b = matched_labels.loc[i, "2.temp"]
        #     print(a)
        #     print(b)
        #     print(matched_labels.loc[i, "2"])
        #     res = tuple(np.concatenate((a, b)))
        #     print(res)
        #     matched_labels.loc[i, "2"] = [res]
        matched_labels = matched_labels.drop("2.temp", axis=1)
    else:
        raise Exception("Case for data types not yet implemented or incompatible")
    return matched_labels

def train(matched_labels: pd.DataFrame,
          model: networks.models.Sequential,
          epochs: int
          ):
    """
    Trains an input model based on previously matched labels and features
    Arguments:
     - matched_labels: previously generated dataframe of matched labels and features(s)
     - model: model to train
     - epochs: number of epochs to train
    """
    if 2 in matched_labels.columns:
        x = np.array(list(matched_labels[2]))
    else:
        x = np.array(list(matched_labels["2"]))
    y = np.array(list(matched_labels["label"]))
    history = model.fit(x=x, y=y, epochs=epochs)
    return history

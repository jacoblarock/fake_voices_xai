from typing import Tuple, Union
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle
import feature_extraction
import networks
import data_container
import mt_operations

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
            to_label = extracted_features["sample"] == name
            extracted_features.loc[to_label, "label"] = label
        if cache:
            with open(cache_path, "wb") as file:
                pickle.dump(extracted_features, file)
    return extracted_features.rename(columns={"feature": name})

# meant for internal use
# expand a one dimensional array into two dimensions
def morph(arr: np.ndarray, vsize: int) -> np.ndarray:
    if len(arr.shape) != 1:
        raise Exception("Vertical size of array to morph must be 1")
    out = np.ndarray((vsize, arr.shape[0]))
    for i in range(vsize):
        out[i] = arr.copy()
    return out

def join_features(matched_labels: pd.DataFrame,
                  feature: pd.DataFrame,
                  feature_name: str
                  ) -> pd.DataFrame:
    if matched_labels["x"][0] == -1 or feature["x"][0] == -1:
        matched_labels = matched_labels.join(feature.set_index(["sample"]), on=["sample"], how="left", rsuffix=".temp")
    else:
        matched_labels = matched_labels.join(feature.set_index(["sample", "x"]), on=["sample", "x"], how="left", rsuffix=".temp")
    matched_labels = matched_labels.reset_index(drop=True)
    for column in matched_labels.columns:
        if column[-5:] != ".temp" and column not in ("sample", "label"):
            column_type = type(matched_labels.loc[0, column])
            if column_type == np.ndarray or column_type == pd.Series:
                sample_shape = matched_labels.loc[0, column].shape
                matched_labels[column] = matched_labels[column].apply(lambda x: x if x is not np.nan else np.ndarray(sample_shape))
            else:
                matched_labels[column] = matched_labels[column].apply(lambda x: x if x is not np.nan else 0)
    matched_labels = matched_labels.drop("y.temp", axis=1)
    matched_labels = matched_labels.rename(columns={"feature": feature_name})
    return matched_labels

def feature_concat(row):
    a = row["feature"].get_underlying()
    b = row["feature.temp"]
    row["feature.temp"] = np.nan
    row["feature"].data = np.concatenate((a, b))
    return row

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
    print("merge_start", str(datetime.now()))
    if type(matched_labels["feature"][0]) == np.ndarray and type(feature["feature"][0]) == np.ndarray:
        # matched_labels = matched_labels.sort_values(by=[0, 1])
        # feature = feature.sort_values(by=[0, 1])
        # morph sizes if number of dimensions is different
        vsize_matched_labels = 1
        if len(matched_labels["feature"][0].shape) > 1:
            vsize_matched_labels = matched_labels["feature"][0].shape[1]
        vsize_feature = 1
        if len(feature["feature"][0].shape) > 1:
            vsize_feature = feature["feature"][0].shape[1]
        if vsize_matched_labels == 1 and vsize_feature != 1:
            matched_labels["feature"] = matched_labels["feature"].apply(morph, args=(vsize_feature,))
        if vsize_matched_labels != 1 and vsize_feature == 1:
            feature["feature"] = feature["feature"].apply(morph, args=(vsize_matched_labels,))
        # perform join and filter
        matched_labels = matched_labels.join(feature.set_index(["sample"]), on=["sample"], how="inner", rsuffix=".temp")
        matched_labels = matched_labels.reset_index(drop=True)
        matched_labels = matched_labels.drop("y.temp", axis=1)
        print("join", str(datetime.now()))
        del feature
        # matched_labels["2"] = matched_labels["2"].apply(data_container.make_container)
        matched_labels["feature"] = mt_operations.apply(matched_labels["feature"], data_container.make_container)
        print("containers made", str(datetime.now()))
        matched_labels["feature", "feature.temp"] = mt_operations.apply(matched_labels["feature", "feature.temp"], feature_concat)
        print("concatted", str(datetime.now()))
        # for i in range(len(matched_labels)):
        #     a = matched_labels.loc[i, "2"].get_underlying()
        #     b = matched_labels.loc[i, "2.temp"]
        #     matched_labels.loc[i, "2.temp"] = np.nan
        #     matched_labels.loc[i, "2"].data = np.concatenate((a, b))
        #     if i % 1000 == 0:
        #         print(i)
        # matched_labels["2"] = matched_labels["2"].apply(data_container.get_underlying)
        matched_labels["feature"] = mt_operations.apply(matched_labels["feature"], data_container.get_underlying)
        print("unpack container", str(datetime.now()))
        #     row = matched_labels.loc[i]
        #     temp = pd.Series({"2": 0})
        #     print(row)
        #     temp.apply(lambda x: np.concatenate((a, b)))
        #     matched_labels.loc[i] = temp
        #     print(matched_labels.loc[i])
        # remaining = matched_labels.join(feature.set_index([0, 1]), on=[0, 1], how="outer", rsuffix=".temp")
        # remaining = pd.concat([matched_labels, remaining]).drop_duplicates(["0", "1"])
        # matched_labels = matched_labels[matched_labels["0"] == matched_labels["0.temp"]]
        # matched_labels = matched_labels.merge(feature, left_on=0, right_on=0, suffixes=("", ".temp"))
        # concat feature in 2 and feature in temp
        # matched_labels["2", "2.temp"] = matched_labels[["2", "2.temp"]].apply(lambda row: np.concatenate((row["2"], row["2.temp"])), axis=1)
        matched_labels = matched_labels.drop("feature.temp", axis=1)
    else:
        raise Exception("Case for data types not yet implemented or incompatible")
    return matched_labels

def gen_batches(indices: pd.Index, batch_size: int) -> list[pd.Index]:
    batches = []
    for i in range(0, len(indices) - batch_size, batch_size):
        batches.append(indices[i:i+batch_size])
    if len(indices) % batch_size != 0:
        batches.append(indices[-(len(indices) % batch_size):])
    return batches

def additive_merge(a: pd.DataFrame,
                   b: pd.DataFrame,
                   ) -> pd.DataFrame:
    a_len = len(a)
    b_len = len(b)
    if b_len > a_len:
        b, a = a, b
        b_len, a_len = a_len, b_len
    if b_len == 0:
        return a
    diff = a_len - b_len
    # if the difference is greater than the length of b, concat b with itself
    while diff >= b_len:
        b = pd.concat((b, b)).reset_index(drop=True)
        b_len = len(b)
        diff = a_len - b_len
    # if the difference is less than the length of b, concat b with a slice of itself
    if 0 < diff < b_len:
        b = pd.concat((b, b.loc[0:diff-1])).reset_index(drop=True)
    return pd.concat((a, b), axis=1)


def train(matched_labels: pd.DataFrame,
          feature_cols: list[str],
          model: networks.models.Sequential,
          epochs: int,
          batch_method: str = "lines",
          batch_size: int = 100000,
          sample_batch_size: int = 100,
          features: list[pd.DataFrame] = [],
          save_as: str | None = None
          ) -> list:
    """
    Trains an input model based on previously matched labels and features
    Arguments:
     - matched_labels: previously generated dataframe of matched labels and features(s)
     - model: model to train
     - epochs: number of epochs to train
    """
    inputs = None
    batches = []
    histories = []
    float_min = np.finfo(np.float64).min
    float_max = np.finfo(np.float64).max
    if batch_method == "lines":
        print("creating batches", datetime.now())
        batches = gen_batches(matched_labels.index, batch_size)
        print("created batches", datetime.now())
        print("begin batch training", datetime.now())
        for batch in batches:
            if len(feature_cols) == 1:
                temp = matched_labels.loc[batch, feature_cols[0]].to_numpy()
                temp = np.stack(temp, axis=0)
                print("start conversion to tensor", datetime.now())
                inputs = tf.convert_to_tensor(temp)
                print("converted to tensor", datetime.now())
            else:
                inputs = []
                for feature in feature_cols:
                    temp = matched_labels.loc[batch, feature].to_numpy()
                    temp = np.stack(temp, axis=0)
                    print("start conversion to tensor", datetime.now())
                    inputs.append(tf.convert_to_tensor(temp))
                    print("converted to tensor", datetime.now())
            labels = tf.convert_to_tensor(matched_labels.loc[batch, "label"].apply(int))
            histories.append(model.fit(x=inputs, y=labels, epochs=epochs))
    if batch_method == "samples":
        if "name" not in matched_labels.columns:
            raise Exception("please use an unmerged labels dataframe")
        progress = 0
        if save_as is not None and os.path.exists("./models/" + save_as + "_progress"):
            with open("./models/" + save_as + "_progress", "rb") as file:
                progress = pickle.load(file)
                print("load progress")
        if save_as is not None and os.path.exists("./models/" + save_as):
            with open("./models/" + save_as, "rb") as file:
                model = pickle.load(file)
                print("load model")
        print(progress)
        samples = matched_labels.loc[progress:].reset_index(drop=True)
        print("\nbegin batch training", datetime.now())
        sample_batches = gen_batches(samples.index, sample_batch_size)
        for sample_batch in sample_batches:
            joined = pd.DataFrame([])
            for col in feature_cols:
                joined[col] = None
            for line in sample_batch:
                sample = samples.loc[line, "name"]
                label = samples.loc[line, "label"]
                print("\ntrain: ", sample, datetime.now())
                print("\nlabel: ", label, datetime.now())
                # joined = pd.DataFrame({"sample": []})
                # joined.loc[0, "sample"] = sample
                # for i in range(len(features)):
                #     joined = joined.join(features[i].set_index(["sample"]), on=["sample"], how="inner")
                #     joined["feature"] = joined["feature"].apply(np.clip, args=(float_min, float_max))
                #     joined = joined.rename(columns={"feature": feature_cols[i]})
                #     joined = joined.drop(columns=["x", "y"])
                new_sample = pd.DataFrame([])
                for i in range(len(features)):
                    temp = features[i].loc[features[i]["sample"] == sample].reset_index(drop=True)
                    temp = pd.DataFrame({feature_cols[i]: temp["feature"]})
                    new_sample = additive_merge(new_sample, temp)
                new_sample["label"] = label
                new_sample = new_sample.reset_index(drop=True)
                joined = pd.concat((joined, new_sample))
            print("\nlen: ", len(joined), datetime.now())
            batches = gen_batches(joined.index, batch_size)
            for batch in batches:
                if len(feature_cols) == 1:
                    temp = joined.loc[batch, feature_cols[0]].to_numpy()
                    temp = np.stack(temp, axis=0)
                    inputs = tf.convert_to_tensor(temp)
                else:
                    inputs = []
                    for feature in feature_cols:
                        temp = joined.loc[batch, feature].to_numpy()
                        temp = np.stack(temp, axis=0)
                        print("start conversion to tensor", datetime.now())
                        inputs.append(tf.convert_to_tensor(temp))
                        print("converted to tensor", datetime.now())
                labels = tf.convert_to_tensor(joined.loc[batch, "label"])
                histories.append(model.fit(x=inputs, y=labels, epochs=epochs))
                if save_as != None:
                    with open("./models/" + save_as, "wb") as file:
                        print("dump model", datetime.now())
                        pickle.dump(model, file)
                        print("model dumped", datetime.now())
                    with open("./models/" + save_as + "_progress", "wb") as file:
                        print("dump progress:", progress)
                        pickle.dump(progress, file)
            progress = progress + 1
    return histories

if __name__ == "__main__":
    a = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]])
    b = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    c = pd.DataFrame([])
    print(additive_merge(a, c))

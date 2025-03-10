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
               name_col: str | int,
               label_col: str | int,
               label_0_val: str | int,
               label_1_val: str | int,
               delimiter: str = ",",
               header: bool = True
               ) -> pd.DataFrame:
    """
    Returns labels from a csv file in a standardized dataframe. this is necessary for efficient label matching.
    Arguments:
    - path: the path of the label file
    - label_col: the name of the column containing labels in the file
    - label_0_val: value in the file of the result 0
    - label_1_val: value in the file of the result 1
    Keyword arguments:
    - delimiter: the delimiter of the file
    - header: bool, True if the file has a header line
    """
    if header:
        data = pd.read_csv(path, delimiter=delimiter)
    else:
        data = pd.read_csv(path, delimiter=delimiter, header=None)
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
    - extracted_features: dataframe of extracted features to match to the labels
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

def _morph(arr: np.ndarray, vsize: int) -> np.ndarray:
    """
    MEANT FOR INTERNAL USE
    Expand a one dimensional array into two dimensions
    """
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
    """
    Joins a dataframe with one or multiple features matched to labels (see match_labels) with another feature dataframe
    This is used for the progressive join feature extraction method
    Arguments:
    - matched_labels: a previously matched label-feature dataframe
    - feature: a new feature dataframe to join with the matched labels
    - feature_name: name of the newly introduced feature to use in the output dataframe
    """
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

def _feature_concat(row):
    """
    MEANT FOR INTERNAL USE
    Use the data_container class to concat two extracted feature arrays into one
    This is a step in the progressive merging method
    Arguments:
    - row: one row of a matched_labels dataframe
    """
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
            matched_labels["feature"] = matched_labels["feature"].apply(_morph, args=(vsize_feature,))
        if vsize_matched_labels != 1 and vsize_feature == 1:
            feature["feature"] = feature["feature"].apply(_morph, args=(vsize_matched_labels,))
        # perform join and filter
        matched_labels = matched_labels.join(feature.set_index(["sample"]), on=["sample"], how="inner", rsuffix=".temp")
        matched_labels = matched_labels.reset_index(drop=True)
        matched_labels = matched_labels.drop("y.temp", axis=1)
        print("join", str(datetime.now()))
        del feature
        # matched_labels["2"] = matched_labels["2"].apply(data_container.make_container)
        matched_labels["feature"] = mt_operations.apply(matched_labels["feature"], data_container.make_container)
        print("containers made", str(datetime.now()))
        matched_labels["feature", "feature.temp"] = mt_operations.apply(matched_labels["feature", "feature.temp"], _feature_concat)
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
    """
    Generates batches of indices based on an input dataframe and a batch size
    Arguments:
    - indices: a pd.Index object from the dataframe to batch
    - batch_size: size of the batches to create
    """
    batches = []
    for i in range(0, len(indices) - batch_size, batch_size):
        batches.append(indices[i:i+batch_size])
    if len(indices) % batch_size != 0:
        batches.append(indices[-(len(indices) % batch_size):])
    return batches

def additive_merge(a: pd.DataFrame | pd.Series,
                   b: pd.DataFrame | pd.Series,
                   ) -> pd.DataFrame:
    """
    Joins two arbitrary dataframes by merging them additively
    MUCH less resource and memory intensive than a cross join without losing any samples
    Arguments:
    - a: first dataframe to merge
    - b: second dataframe to merge
    """
    a_len = len(a)
    b_len = len(b)
    if b_len > a_len:
        b, a = a, b
        b_len, a_len = a_len, b_len
    if b_len == 0:
        return pd.DataFrame(a)
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
          batch_method: str = "samples",
          batch_size: int = 100000,
          sample_batch_size: int = 100,
          features: list[pd.DataFrame] = [],
          validation_split: float = 0,
          save_as: str | None = None
          ) -> list:
    """
    Trains an input model based on previously matched labels and features
    Arguments:
    - matched_labels: previously generated dataframe of matched labels and features(s)
    - feature_cols: list of the names of the features in either the matched_labels or features dataframe
    - model: model to train
    - epochs: number of epochs to train
    Keyword arguments:
    - batch_method: either "samples" or "lines"
      "lines" will expect a matched_labels dataframe containing labels and features and iterate in batched over the lines of this dataframe
      "samples" will expect on dataframe with only the labels (unmatched) and a list of dataframes passed through the features kwarg.
      Batches with this method are file-based and created with the additive join instead of a left-join, resulting in no potential loss and infill with zeroes
    - batch_size: MAXIMUM batch size; used for both batch methods
    - sample_batch_size: ONLY for the "samples" batch method; number of files to include in one sample-based batch
      batch_size will come into effect with this method only when the result of the additive join on the file batch is longer than batch_size
    - features: a list of dataframes containing extracted features for use with the "samples" batch method
    - validation_split: float between zero and one, the portion of the training batch to use for validation. Default is 0 -> no validation during batch training
    - save_as: path to save the model under after each training batch
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
            histories.append(model.fit(x=inputs, y=labels, epochs=epochs, validation_split=validation_split))
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
            print("creating batch")
            for line in sample_batch:
                sample = samples.loc[line, "name"]
                label = samples.loc[line, "label"]
                print("\r", " " * 40, "\r", end="", flush=True)
                print("adding: ", sample, "label: ", label, datetime.now(), end="", flush=True)
                new_sample = pd.DataFrame([])
                for i in range(len(features)):
                    temp = features[i].loc[features[i]["sample"] == sample].reset_index(drop=True)
                    temp = pd.DataFrame({feature_cols[i]: temp["feature"]})
                    new_sample = additive_merge(new_sample, temp)
                new_sample["label"] = label
                new_sample = new_sample.reset_index(drop=True)
                joined = pd.concat((joined, new_sample))
            print("len: ", len(joined), datetime.now())
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
                        print("\r", " " * 40, "\r", end="", flush=True)
                        print("start conversion to tensor", datetime.now(), end="", flush=True)
                        inputs.append(tf.convert_to_tensor(temp))
                        del temp
                        print("\r", " " * 40, "\r", end="", flush=True)
                        print("converted to tensor", datetime.now(), end="", flush=True)
                labels = tf.convert_to_tensor(joined.loc[batch, "label"])
                histories.append(model.fit(x=inputs, y=labels, epochs=epochs, validation_split=validation_split))
                if save_as != None:
                    with open("./models/" + save_as, "wb") as file:
                        print("dump model", datetime.now())
                        pickle.dump(model, file)
                        print("model dumped", datetime.now())
                    with open("./models/" + save_as + "_progress", "wb") as file:
                        print("dump progress:", progress+sample_batch_size)
                        pickle.dump(progress+sample_batch_size, file)
                    with open("./models/" + save_as + "_histories", "wb") as file:
                        print("dump histories")
                        pickle.dump(histories, file)
            progress = progress + sample_batch_size
    return histories

# TODO: lines batch method
# def evaluate(labels: pd.DataFrame,
#               feature_cols: list[str],
#               features: list[pd.DataFrame],
#               model: networks.models.Sequential,
#               batch_size: int = 100000,
#               sample_batch_size: int = 100,
#               batch_method: str = "samples"
#               ) -> list:
#     results = []
#     if batch_method == "samples":
#         if "name" not in labels.columns:
#             raise Exception("please use an unmerged labels dataframe")
#         samples = labels.reset_index(drop=True)
#         print("\nbegin batch eval", datetime.now())
#         sample_batches = gen_batches(samples.index, sample_batch_size)
#         for sample_batch in sample_batches:
#             joined = pd.DataFrame([])
#             for col in feature_cols:
#                 joined[col] = None
#             for line in sample_batch:
#                 sample = samples.loc[line, "name"]
#                 label = samples.loc[line, "label"]
#                 print("\neval: ", sample, datetime.now())
#                 print("\nlabel: ", label, datetime.now())
#                 new_sample = pd.DataFrame([])
#                 for i in range(len(features)):
#                     temp = features[i].loc[features[i]["sample"] == sample].reset_index(drop=True)
#                     temp = pd.DataFrame({feature_cols[i]: temp["feature"]})
#                     new_sample = additive_merge(new_sample, temp)
#                 new_sample["label"] = label
#                 new_sample = new_sample.reset_index(drop=True)
#                 joined = pd.concat((joined, new_sample))
#             print("\nlen: ", len(joined), datetime.now())
#             batches = gen_batches(joined.index, batch_size)
#             for batch in batches:
#                 if len(feature_cols) == 1:
#                     temp = joined.loc[batch, feature_cols[0]].to_numpy()
#                     temp = np.stack(temp, axis=0)
#                     inputs = tf.convert_to_tensor(temp)
#                 else:
#                     inputs = []
#                     for feature in feature_cols:
#                         temp = joined.loc[batch, feature].to_numpy()
#                         temp = np.stack(temp, axis=0)
#                         print("start conversion to tensor", datetime.now())
#                         inputs.append(tf.convert_to_tensor(temp))
#                         print("converted to tensor", datetime.now())
#                 labels = tf.convert_to_tensor(joined.loc[batch, "label"])
#                 results.append(model.evaluate(x=inputs, y=labels))
#     return results

def isolate_sample(features: list[pd.DataFrame],
                   sample: str
                   ) -> list[pd.DataFrame]:
    """
    Isolates the features of a single sample out of dataframes of extracted features of many
    samples.
    Arguments:
    - features: list of feature dataframes
    - sample: name of the sample to isolate (including file extension)
    """
    out = []
    for i in range(len(features)):
        out.append(features[i].loc[features[i]["sample"] == sample].reset_index(drop=True))
    return out

def classify(model: networks.models.Sequential,
             features: list[pd.DataFrame],
             feature_cols: list[str]
             ) -> np.ndarray:
    """
    Classifies an INDIVIDUAL sample based on an input model and features
    Arguments:
    - model: TRAINED model to use for classification
    - features: a list of dataframes features extracted from the sample
    - feature_cols: the names of the features in the dataframes of features
    """
    joined = pd.DataFrame([])
    for i in range(len(features)):
        temp = pd.DataFrame({feature_cols[i]: features[i]["feature"]})
        joined = additive_merge(joined, temp)
    if len(feature_cols) == 1:
        temp = joined.loc[joined.index, feature_cols[0]].to_numpy()
        temp = np.stack(temp, axis=0)
        inputs = tf.convert_to_tensor(temp)
    else:
        inputs = []
        for feature in feature_cols:
            temp = joined.loc[joined.index, feature].to_numpy()
            temp = np.stack(temp, axis=0)
            inputs.append(tf.convert_to_tensor(temp))
    return model.predict(inputs)

def evaluate(labels: pd.DataFrame,
             feature_cols: list[str],
             features: list[pd.DataFrame],
             model: networks.models.Sequential,
             threshold: float = 0.5,
             summary_method: str = "average",
             save_as: str | None = None
             ) -> Tuple[list, dict]:
    """
    Classifies a directory of samples and returns summary statistics on model accuracy
    Arguments:
    - labels: dataframe of labels to use for prediction evaluation
    - feature_cols: list of names of features in the features list
    - features: list of dataframes of extracted features
    - model: model to evaluate
    Keyword arguments:
    - threshold: threshold for classification: below the threshold is classified as 0, above is 1
    - summary_method: summarize the results of the classifications using either the "average" (statistical mean) or the "median"
    """
    summary = {"tp": 0,
           "tn": 0,
           "fp": 0,
           "fn": 0}
    results = []
    for i in labels.index:
        sample = labels.loc[i, "name"]
        label = labels.loc[i, "label"]
        sample_features = []
        for i in range(len(features)):
            sample_features.append(features[i].loc[features[i]["sample"] == sample].reset_index(drop=True))
        res_arr = classify(model, sample_features, feature_cols)
        summary_func = np.average
        if summary_method == "median":
            summary_func = np.median
        result = summary_func(res_arr)
        results.append({"result": result, "label": label})
        result = 0 if result < threshold else 1
        if label == 0:
            if result == 0:
                summary["tn"] += 1
            if result == 1:
                summary["fp"] += 1
        if label == 1:
            if result == 0:
                summary["fn"] += 1
            if result == 1:
                summary["tp"] += 1
        print(summary)
    return results, summary

if __name__ == "__main__":
    pass

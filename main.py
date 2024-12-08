import feature_extraction
import classification
import networks
import mt_operations
import explainers
import metrics
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
from datetime import datetime
from keras._tf_keras.keras import utils
import numpy as np
import pickle
import os
print("dependencies loaded")

def plot_1d(data_arr):
    plt.plot(data_arr)
    plt.show()
    return

def plot_2d(data_arr):
    plt.imshow(data_arr, interpolation="nearest")
    plt.show()
    return

def extract_progressive_merging(labels, dataset_dir, dataset_ext, extraction_kwargs):
    # generate hnrs
    hnrs = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_hnrs,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="hnrs"
                                   )
    print("hnrs extracted", datetime.now())
    print("shape", hnrs.shape)
    print("max x:", max(hnrs["x"]))

    # match labels to the feature
    matched_labels = classification.match_labels(labels, hnrs, "hnrs")
    del hnrs
    print("matched labels", datetime.now())

    # generate mel specs
    mel_spec = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.gen_mel_spec,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="mel_spec"
                                   )
    print("mel extracted", datetime.now())
    print("shape", mel_spec.shape)
    print("max x:", max(mel_spec["x"]))

    # merge features
    matched_labels = classification.join_features(matched_labels, mel_spec, "mel_spec")
    del mel_spec
    print("merged shape", matched_labels.shape)

    # generate mfccs
    mfccs = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.gen_mfcc,
                                         [30]],
                                   kwargs=extraction_kwargs,
                                   cache_name="mfccs"
                                   )
    print("mfccs extracted", datetime.now())
    print("shape", mfccs.shape)
    print("max x:", max(mfccs["x"]))

    # merge features
    matched_labels = classification.join_features(matched_labels, mfccs, "mfccs")
    del mfccs
    print("merged shape", matched_labels.shape)

    # generate hnrs
    f0_lens = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_f0_lens,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="f0_lens"
                                   )
    print("f0_lens extracted", datetime.now())
    print("shape", f0_lens.shape)
    print("max x:", max(f0_lens["x"]))

    # merge features
    matched_labels = classification.join_features(matched_labels, f0_lens, "f0_lens")
    del f0_lens
    print("merged shape", matched_labels.shape)
    print("joined", datetime.now())
    # matched_labels["hnrs"] = matched_labels["hnrs"].apply(classification.morph, vsize=30)
    # print("morph", datetime.now())

    return (["hnrs", "mel_spec", "mfccs", "f0_lens"], matched_labels)

def extract_separate(dataset_dir, dataset_ext, extraction_kwargs) -> Tuple[list[str], list[pd.DataFrame]]:
    hnrs = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_hnrs,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="hnrs"
                                   )
    print("hnrs extracted", datetime.now())
    print("shape", hnrs.shape)
    print("max x:", max(hnrs["x"]))

    mel_spec = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.gen_mel_spec,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="mel_spec"
                                   )
    print("mel extracted", datetime.now())
    print("shape", mel_spec.shape)
    print("max x:", max(mel_spec["x"]))

    mfccs = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.gen_mfcc,
                                         [30]],
                                   kwargs=extraction_kwargs,
                                   cache_name="mfccs"
                                   )
    print("mfccs extracted", datetime.now())
    print("shape", mfccs.shape)
    print("max x:", max(mfccs["x"]))

    f0_lens = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_f0_lens,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="f0_lens"
                                   )
    print("f0_lens extracted", datetime.now())
    print("shape", f0_lens.shape)
    print("max x:", max(f0_lens["x"]))

    onset_strength = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_onset_strength,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="onset_strength"
                                   )
    print("onset_strength extracted", datetime.now())
    print("shape", onset_strength.shape)
    print("max x:", max(onset_strength["x"]))

    intensity = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_intensity,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="intensity"
                                   )
    print("intensity extracted", datetime.now())
    print("shape", intensity.shape)
    print("max x:", max(intensity["x"]))

    pitch_flucs = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_pitch_fluctuation,
                                         [1]],
                                   kwargs=extraction_kwargs,
                                   cache_name="pitch_flucs"
                                   )
    print("pitch_flucs extracted", datetime.now())
    print("shape", pitch_flucs.shape)
    print("max x:", max(pitch_flucs["x"]))

    local_jitter = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_local_jitter,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="local_jitter"
                                   )
    print("local_jitter extracted", datetime.now())
    print("shape", local_jitter.shape)
    print("max x:", max(local_jitter["x"]))

    rap_jitter = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_rap_jitter,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="rap_jitter"
                                   )
    print("rap_jitter extracted", datetime.now())
    print("shape", rap_jitter.shape)
    print("max x:", max(rap_jitter["x"]))

    ppq5_jitter = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_ppq5_jitter,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="ppq5_jitter"
                                   )
    print("ppq5_jitter extracted", datetime.now())
    print("shape", ppq5_jitter.shape)
    print("max x:", max(ppq5_jitter["x"]))

    ppq55_jitter = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_ppq55_jitter,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="ppq55_jitter"
                                   )
    print("ppq55_jitter extracted", datetime.now())
    print("shape", ppq55_jitter.shape)
    print("max x:", max(ppq55_jitter["x"]))

    local_shim = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_shim_local,
                                         []],
                                   kwargs=extraction_kwargs,
                                   cache_name="local_shim"
                                   )
    print("local_shim extracted", datetime.now())
    print("shape", local_shim.shape)
    print("max x:", max(local_shim["x"]))

    rap_shim = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_shim_apqx,
                                         [3]],
                                   kwargs=extraction_kwargs,
                                   cache_name="rap_shim"
                                   )
    print("rap_shim extracted", datetime.now())
    print("shape", rap_shim.shape)
    print("max x:", max(rap_shim["x"]))

    ppq5_shim = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_shim_apqx,
                                         [5]],
                                   kwargs=extraction_kwargs,
                                   cache_name="ppq5_shim"
                                   )
    print("ppq5_shim extracted", datetime.now())
    print("shape", ppq5_shim.shape)
    print("max x:", max(ppq5_shim["x"]))

    ppq55_shim = mt_operations.file_func(feature_extraction.bulk_extract,
                                   dataset_dir,
                                   args=[dataset_dir,
                                         dataset_ext,
                                         feature_extraction.get_shim_apqx,
                                         [55]],
                                   kwargs=extraction_kwargs,
                                   cache_name="ppq55_shim"
                                   )
    print("ppq55_shim extracted", datetime.now())
    print("shape", ppq55_shim.shape)
    print("max x:", max(ppq55_shim["x"]))

    return (["hnrs",
             "mel_spec",
             "mfccs",
             "f0_lens",
             "onset_strength",
             "intensity",
             "pitch_flucs",
             "local_jitter",
             "rap_jitter",
             "ppq5_jitter",
             "ppq55_jitter",
             "local_shim",
             "rap_shim",
             "ppq5_shim",
             "ppq55_shim"],
            [hnrs,
             mel_spec,
             mfccs,
             f0_lens,
             onset_strength,
             intensity,
             pitch_flucs,
             local_jitter,
             rap_jitter,
             ppq5_jitter,
             ppq55_jitter,
             local_shim,
             rap_shim,
             ppq5_shim,
             ppq55_shim])

def eval(model: str | classification.networks.models.Sequential, eval_from: int):

    dataset_dir = "./datasets/release_in_the_wild/"
    dataset_ext = "wav"
    extraction_kwargs={"cache": False,
            "use_cached": False,
            "window_length": 30,
            "window_height": 30
            }
    # filter for the samples to evaluate
    # filter = pd.DataFrame({"name": list(os.listdir(dataset_dir))})
    filter = pd.DataFrame({"name": list(range(eval_from, 31778))})
    # If necessary, change the filter to str and append file endings
    filter["name"] = filter["name"].apply(lambda x : str(x) + "." + dataset_ext)

    # load labels
    labels = classification.get_labels("./datasets/release_in_the_wild/meta.csv", "file", "label", "spoof", "bona-fide")
    # labels["name"] = labels["name"].apply(lambda x: x + ".flac")
    labels = labels.join(filter.set_index("name"), how="inner", on=["name"])

    # load model if a path is provided
    if type(model) == str:
        model = pickle.load(open(model, "rb"))
    print(model.summary())

    # creates a list of dataframes for each extracted feature for sample-based batching
    feature_names, features = extract_separate(dataset_dir, dataset_ext, extraction_kwargs)

    results, summary = classification.evaluate(labels, feature_names, features, model)
    print(summary)
    with open("cache/results.txt", "w") as file:
        file.write(str(results))
    with open("cache/results_summary.txt", "w") as file:
        file.write(str(summary))

def train(eval_until: int):
    feature_extraction.check_cache()

    dataset_dir = "./datasets/release_in_the_wild"
    dataset_ext = "wav"
    extraction_kwargs={"cache": False,
            "use_cached": False,
            "window_length": 30,
            "window_height": 30
            }

    filter = pd.DataFrame({"name": list(range(eval_until))})
    # If necessary, change the filter to str and append file endings
    filter["name"] = filter["name"].apply(lambda x : str(x) + "." + dataset_ext)

    # get labels
    labels = classification.get_labels("./datasets/release_in_the_wild/meta.csv", "file", "label", "spoof", "bona-fide")
    labels = labels.join(filter.set_index("name"), how="inner", on=["name"])
    print("labels", datetime.now())

    # create one large dataframe with all features labelled for training
    # training batches made from a set number of lines
    # feature_names, matched_labels = extract_progressive_merging()

    # creates a list of dataframes for each extracted feature for sample-based batching
    feature_names, features = extract_separate(dataset_dir, dataset_ext, extraction_kwargs)

    # create and train the model
    hnr_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30, name="hnrs")
    mel_model = networks.create_cnn_2d((30, 30), 32, 3, pooling=True, output_size=30, name="mel")
    mfcc_model = networks.create_cnn_2d((30, 30), 32, 3, pooling=True, output_size=30, name="mfcc")
    f0_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30, name="f0_lens")
    onset_strength_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30, name="onset_strength")
    intensity_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30, name="intensity")
    pitch_fluc_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30, name="pitch_flucs")
    local_jitter_model = networks.single_input(name="local_jitter")
    rap_jitter_model = networks.single_input(name="rap_jitter")
    ppq5_jitter_model = networks.single_input(name="ppq5_jitter")
    ppq55_jitter_model = networks.single_input(name="ppq55_jitter")
    local_shim_model = networks.single_input(name="local_shimmer")
    rap_shim_model = networks.single_input(name="rap_shimmer")
    ppq5_shim_model = networks.single_input(name="ppq5_shimmer")
    ppq55_shim_model = networks.single_input(name="ppq55_shimmer")
    model = networks.stitch_and_terminate([hnr_model,
                                           mel_model,
                                           mfcc_model,
                                           f0_model,
                                           onset_strength_model,
                                           intensity_model,
                                           pitch_fluc_model,
                                           local_jitter_model,
                                           rap_jitter_model,
                                           ppq5_jitter_model,
                                           ppq55_jitter_model,
                                           local_shim_model,
                                           rap_shim_model,
                                           ppq5_shim_model,
                                           ppq55_shim_model])
    print(model.summary())
    try:
        utils.plot_model(model, "model_plot.png", show_layer_names=True)
    except:
        print("model plot not possible")
    # histories = classification.train(matched_labels, feature_names, model, 3, batch_size=100000)
    histories = classification.train(labels, feature_names, model, 1, batch_size=1000000, features=features, batch_method="samples", save_as="test_multi_percep")
    for history in histories:
        print(history)

def classify_test(model: str | classification.networks.models.Sequential, filename: str):

    dataset_dir = "./datasets/release_in_the_wild"
    dataset_ext = "wav"

    # load model if a path is provided
    if type(model) == str:
        model = pickle.load(open(model, "rb"))
    print(model.summary())

    hnrs = feature_extraction.bulk_extract(dataset_dir,
                                           dataset_ext,
                                           feature_extraction.get_hnrs,
                                           [],
                                           file_list=[filename],
                                           window_length=30,
                                           window_height=30,
                                           cache=False,
                                           use_cached=False)

    mel_spec = feature_extraction.bulk_extract(dataset_dir,
                                           dataset_ext,
                                           feature_extraction.gen_mel_spec,
                                           [],
                                           file_list=[filename],
                                           window_length=30,
                                           window_height=30,
                                           cache=False,
                                           use_cached=False)

    mfccs = feature_extraction.bulk_extract(dataset_dir,
                                           dataset_ext,
                                           feature_extraction.gen_mfcc,
                                           [30],
                                           file_list=[filename],
                                           window_length=30,
                                           window_height=30,
                                           cache=False,
                                           use_cached=False)

    f0_lens = feature_extraction.bulk_extract(dataset_dir,
                                           dataset_ext,
                                           feature_extraction.get_f0_lens,
                                           [],
                                           file_list=[filename],
                                           window_length=30,
                                           window_height=30,
                                           cache=False,
                                           use_cached=False)

    results = classification.classify(model, [hnrs, mel_spec, mfccs, f0_lens], ["hnrs", "mel_spec", "mfccs", "f0_lens"])
    avg = np.average(results)
    median = np.median(results)
    return (avg, median)

def explainer_test(model):
    dataset_dir = "./datasets/release_in_the_wild/"
    dataset_ext = "wav"
    extraction_kwargs={"cache": False,
            "use_cached": False,
            "window_length": 30,
            "window_height": 30
            }

    feature_cols, features = extract_separate(dataset_dir, dataset_ext, extraction_kwargs)

    if type(model) == str:
        model = pickle.load(open(model, "rb"))
    print(model.summary())

    inter_data = explainers.gen_intermediate_train_data(model, features, feature_cols, 1000000)
    inter_data = explainers.inter_data_concat(inter_data)
    print("train data concatted", classification.datetime.now())
    exp = explainers.explain(model, features, feature_cols, inter_data)
    print(exp.as_list())

if __name__ == "__main__":
    """
    More specific parameters are in the extraction, train and eval functions, such as dataset directory.
    """
    # train(10000)
    # eval("models/test_multi_percep", 10000)
    explainer_test("./trained_models/ItW_multi_percep_until10000")

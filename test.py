import feature_extraction
import classification
import networks
import mt_operations
import matplotlib.pyplot as plt
from datetime import datetime
print("dependencies loaded")

def plot_1d(data_arr):
    plt.plot(data_arr)
    plt.show()

def plot_2d(data_arr):
    plt.imshow(data_arr, interpolation="nearest")
    plt.show()

if __name__ == "__main__":
    feature_extraction.check_cache()

    # generate hnrs
    # hnrs = feature_extraction.bulk_extract("./datasets/release_in_the_wild", "wav", feature_extraction.get_hnrs, [])
    hnrs = mt_operations.file_func(feature_extraction.bulk_extract,
                                   "./datasets/release_in_the_wild",
                                   args=["./datasets/release_in_the_wild",
                                         "wav",
                                         feature_extraction.get_hnrs,
                                         []],
                                   kwargs={"cache": False,
                                           "use_cached": False,
                                           "window_length": 30,
                                           "window_height": 30},
                                   cache_name="hnrs"
                                   )
    print("hnrs extracted", datetime.now())

    # generate mel specs
    # mel_spec = feature_extraction.bulk_extract("./datasets/release_in_the_wild", "wav", feature_extraction.gen_mel_spec, [])
    mel_spec = mt_operations.file_func(feature_extraction.bulk_extract,
                                   "./datasets/release_in_the_wild",
                                   args=["./datasets/release_in_the_wild",
                                         "wav",
                                         feature_extraction.gen_mel_spec,
                                         []],
                                   kwargs={"cache": False,
                                           "use_cached": False,
                                           "window_length": 30,
                                           "window_height": 30},
                                   cache_name="mel_spec"
                                   )
    print("mel extracted", datetime.now())

    # label and merge the features
    labels = classification.get_labels("./datasets/release_in_the_wild/meta.csv", "file", "label", "spoof", "bona-fide")
    print("labels", datetime.now())
    matched_labels = classification.match_labels(labels, hnrs, "hnrs")
    del hnrs
    print("matched labels", datetime.now())
    matched_labels = classification.join_features(matched_labels, mel_spec, "mel_spec")
    del mel_spec
    print("joined", datetime.now())
    matched_labels["hnrs"] = matched_labels["hnrs"].apply(classification.morph, vsize=30)
    print("morph", datetime.now())
    print(matched_labels)
    print(matched_labels.shape)

    # create and train the model
    hnr_model = networks.create_cnn_2d((30, 30), 32, 2, pooling=False)
    mel_model = networks.create_cnn_2d((30, 30), 32, 2, pooling=False)
    model = networks.stitch_and_terminate([hnr_model, mel_model])
    print(model.summary())
    history = classification.train(matched_labels, ["hnrs", "mel_spec"], model, 3)
    print(history)

import feature_extraction
import classification
import networks
import mt_operations
import matplotlib.pyplot as plt
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
    print("hnrs extracted")

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
    print("mel extracted")

    # label and merge the features
    labels = classification.get_labels("./datasets/release_in_the_wild/meta.csv", "file", "label", "spoof", "bona-fide")
    print("labels")
    matched_labels = classification.match_labels(labels, hnrs, "hnrs")
    print("matched labels")
    matched_labels = classification.merge(matched_labels, mel_spec)
    print("merged")
    print(matched_labels)
    print(matched_labels.shape)
    print(matched_labels["2"][0].shape)

    # create and train the model
    model = networks.create_cnn_2d((60, 30), 32, 2, pooling=False)
    print(model.summary())
    history = classification.train(matched_labels, model, 10)
    print(history.history)

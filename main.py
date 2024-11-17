import feature_extraction
import classification
import networks
import mt_operations
import matplotlib.pyplot as plt
from datetime import datetime
from keras._tf_keras.keras import utils
print("dependencies loaded")

def plot_1d(data_arr):
    plt.plot(data_arr)
    plt.show()

def plot_2d(data_arr):
    plt.imshow(data_arr, interpolation="nearest")
    plt.show()

if __name__ == "__main__":
    feature_extraction.check_cache()

    dataset_dir = "./datasets/release_in_the_wild"
    dataset_ext = "wav"
    extraction_kwargs={"cache": False,
            "use_cached": False,
            "window_length": 30,
            "window_height": 30}

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
    print(hnrs)

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
    print(mel_spec)

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
    print(mfccs)

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
    print(hnrs)

    # label and merge the features
    labels = classification.get_labels("./datasets/release_in_the_wild/meta.csv", "file", "label", "spoof", "bona-fide")
    print("labels", datetime.now())
    matched_labels = classification.match_labels(labels, hnrs, "hnrs")
    print(matched_labels)
    del hnrs
    print("matched labels", datetime.now())
    matched_labels = classification.join_features(matched_labels, mel_spec, "mel_spec")
    del mel_spec
    matched_labels = classification.join_features(matched_labels, mfccs, "mfccs")
    del mfccs
    matched_labels = classification.join_features(matched_labels, f0_lens, "mfccs")
    del f0_lens
    print("joined", datetime.now())
    # matched_labels["hnrs"] = matched_labels["hnrs"].apply(classification.morph, vsize=30)
    # print("morph", datetime.now())
    print(matched_labels)
    print(matched_labels.shape)

    # create and train the model
    hnr_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30)
    mel_model = networks.create_cnn_2d((30, 30), 32, 3, pooling=True, output_size=30)
    mfcc_model = networks.create_cnn_2d((30, 30), 32, 3, pooling=True, output_size=30)
    f0_model = networks.create_cnn_1d(30, 32, 3, pooling=False, output_size=30)
    model = networks.stitch_and_terminate([hnr_model, mel_model, mfcc_model, f0_model])
    print(model.summary())
    try:
        utils.plot_model(model, "model_plot.png", show_shapes=True)
    except:
        print("model plot not possible")
    histories = classification.train(matched_labels, ["hnrs", "mel_spec", "mfccs", "f0_lens"], model, 3, batch_size=100000)
    print(histories)

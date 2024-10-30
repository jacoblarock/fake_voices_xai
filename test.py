import feature_extraction
import matplotlib.pyplot as plt
import classification
print("dependencies loaded")

def plot_1d(data_arr):
    plt.plot(data_arr)
    plt.show()

def plot_2d(data_arr):
    plt.imshow(data_arr, interpolation="nearest")
    plt.show()

if __name__ == "__main__":
    hnrs = feature_extraction.bulk_extract("./datasets/release_in_the_wild", "wav", feature_extraction.get_hnrs, [])
    print("hnrs extracted")
    labels = classification.get_labels("./datasets/release_in_the_wild/meta.csv", "file", "label", "spoof", "bona-fide")
    print("labels")
    extracted_features = classification.match_labels(labels, hnrs, "hnrs")
    print(extracted_features)

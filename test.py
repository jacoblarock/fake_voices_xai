import feature_extraction
import matplotlib.pyplot as plt
print("dependencies loaded")

def plot_1d(data_arr):
    plt.plot(data_arr)
    plt.show()

def plot_2d(data_arr):
    plt.imshow(data_arr, interpolation="nearest")
    plt.show()

if __name__ == "__main__":
    filepath = "release_in_the_wild/0.wav"
    samples, sample_rate = feature_extraction.lr_load_file(filepath)
    hnrs = feature_extraction.get_hnrs(samples, sample_rate)
    print(len(samples), len(hnrs))
    filepath = "release_in_the_wild/1.wav"
    samples, sample_rate = feature_extraction.lr_load_file(filepath)
    hnrs = feature_extraction.get_hnrs(samples, sample_rate)
    print(len(samples), len(hnrs))

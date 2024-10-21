import feature_extraction
import numpy as np
import librosa as lr
print("dependencies loaded")

if __name__ == "__main__":
    filepath = "test.mp3"
    samples, sample_rate = feature_extraction.lr_load_file(filepath)
    flucs = feature_extraction.get_pitch_fluctuation(samples, sample_rate, 5)
    print(feature_extraction.get_summary_stats(flucs))

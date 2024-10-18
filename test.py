import feature_extraction
import numpy as np
import librosa as lr
print("dependencies loaded")

if __name__ == "__main__":
    filepath = "test.mp3"
    samples, sample_rate = feature_extraction.lr_load_file(filepath)
    print(samples.shape)
    pitches = feature_extraction.get_pitches(samples, sample_rate)
    print(pitches)
    fluctuations = feature_extraction.get_pitch_fluctuation(samples, sample_rate, 3)
    print(fluctuations)

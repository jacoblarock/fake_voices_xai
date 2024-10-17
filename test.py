import feature_extraction
import numpy as np
print("dependencies loaded")

if __name__ == "__main__":
    filepath = "test.mp3"
    samples, sample_rate = feature_extraction.lr_load_file(filepath)
    print(feature_extraction.get_local_jitter(samples, sample_rate))
    print(feature_extraction.get_rap_jitter(samples, sample_rate))
    print(feature_extraction.get_ppq5_jitter(samples, sample_rate))
    print(feature_extraction.get_ppq55_jitter(samples, sample_rate))
    print(feature_extraction.get_shim_local(samples, sample_rate))
    print(feature_extraction.get_shim_apqx(samples, sample_rate, 20))

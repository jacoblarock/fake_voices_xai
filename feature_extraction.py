import sliding_window
from typing import Callable, Tuple, Union
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

universal_fmin = 65
universal_fmax = 2093

def lr_load_file(filepath: str) -> Tuple[np.ndarray, Union[int, float]]:
    """
    Loads a from a given path and returns samples and sample rate
    Performs automatic downsampling if necessary (large audio files)
    """
    sr = lr.get_samplerate(filepath)
    while True:
        try:
            samples, _ = lr.load(filepath, sr=sr)
            break
        except Exception as e:
            print(e)
            sr = int(sr * 0.8)
            print("NP Array too large, downsampling audio to: " + str(sr))
    return samples, sr

def check_cache() -> int:
    cache_paths = ["./cache",
                   "./cache/extracted_features",
                   "./cache/matched_labels"
                   "./cache/mt_ops"
                   ]
    complete = 1
    for path in cache_paths:
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                return -1
    return 0

# TODO: MULTITHREADING!!!
def bulk_extract(directory: str,
                 extension: str,
                 feature: Callable,
                 args: list,
                 file_list: list[str] = [],
                 window_length: int = 10,
                 window_height: int = 10,
                 summarize: bool = False,
                 cache=True,
                 use_cached=True
                 ) -> pd.DataFrame:
    """
    Apply a given feature extractor to a directory of files.
    Positional arguments:
    - directory: directory path to search in
    - extension: extension of the files (for example, ".wav")
    - feature: Callable to apply to the audio files
    - args: arguments for the feature extractor function, if necessary, for example gen_mfcc expects a "count" argument.
    Keyword arguments:
    - summarize: whether the results (if results are np.ndarray) should be summarizes into quantiles (default False)
    - cache: whether the results should be cached (default True)
    - use_cached: whether previously cached results should be returned (default True)
      NOTE: cache=True and use_cached=False will overwrite an existing cache, if one exists
    """
    if cache or use_cached:
        check_cache()
    if directory[-1] != "/":
        directory = directory + "/"
    cache_path = "./cache/extracted_features/" + directory[:-1].split("/")[-1] + "_" + feature.__name__ + ("_sum" if summarize else "")
    out = []
    if os.path.isfile(cache_path) and use_cached:
        with open(cache_path, "rb") as file:
            out = pickle.load(file)
    else:
        if len(file_list) == 0:
            files = os.listdir(directory)
        else:
            files = file_list
        for file in files:
            if file[-len(extension):] != extension:
                files.remove(file)
        for file in files:
            samples, sample_rate = lr_load_file(directory + file)
            if not summarize:
                extracted_feature = feature(samples, sample_rate, *args)
                if type(extracted_feature) == np.ndarray:
                    window = sliding_window.window(extracted_feature, window_length, window_height)
                    state = 1
                    out.append((file, window.x, window.y, window.get_window()))
                    while state == 1:
                        state = window.smart_hop(0.5)
                        out.append((file, window.x, window.y, window.get_window()))
                else:
                    out.append((file, -1, -1, extracted_feature))
            if summarize:
                out.append((file, 0, get_summary_stats(feature(samples, sample_rate, *args))))
        if cache:
            with open(cache_path, "wb") as file:
                pickle.dump(pd.DataFrame(out), file)
    out = pd.DataFrame(out)
    out = out.rename(columns={0: "sample",
                              1: "x",
                              2: "y",
                              3: "feature"})
    return out

# Summarize the results of a calculation resulting in an array into quartiles
# How does this affect classification?
def get_summary_stats(data: np.ndarray) -> dict:
    data = data[data != 0.0]
    out = {}
    for i in range(1, 10):
        out[i/10] = np.quantile(data, i/10)
    return out

# -----
# Below are individual feature extraction functions
# -----

def gen_mel_spec(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    # Short-time Fourier transform 
    sgram = lr.stft(samples)
    sgram_mag, sgram_phase = lr.magphase(sgram)
    # return mel_spec powers
    return lr.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

# Imperceptible features
def gen_stft_mags(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    stft = lr.stft(samples)
    mags, _ =  lr.magphase(stft)
    return mags

def gen_mfcc(samples: np.ndarray, sample_rate: int | float, count: int) -> np.ndarray:
    mel_spec = gen_mel_spec(samples, sample_rate)
    return lr.feature.mfcc(S=lr.power_to_db(mel_spec), n_mfcc=20)

# Get the lengths of every fundamental frequency cycle
def get_f0_lens(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    f0 = lr.yin(samples, fmin=universal_fmin, fmax=universal_fmax)
    return 1 / f0

# Perceptible features from Chaiwongyen et al, 2023:
# Jitter compared to next neighbor sample
def get_local_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    rolled_f0 = np.roll(f0_lens, -1)
    difsum = sum(np.abs(f0_lens[:-1] - rolled_f0[:-1]))
    del rolled_f0
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

# Jitter compared to both neighboring samples
def get_rap_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    difsum = 0
    for i in range(1, N - 1):
        difsum += np.abs(f0_lens[i] - np.average(f0_lens[i-1:i+2]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

# Jitter compared to nearest five neighbors
def get_ppq5_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    difsum = 0
    for i in range(2, N - 2):
        difsum += np.abs(f0_lens[i] - np.average(f0_lens[i-2:i+3]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

# Jitter compared to nearest fifty five neighbors
def get_ppq55_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    difsum = 0
    for i in range(27, N - 27):
        difsum += np.abs(f0_lens[i] - np.average(f0_lens[i-27:i+28]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

# Shimmer compared to next neighbor sample
def get_shim_local(samples: np.ndarray, sample_rate: int | float) -> float:
    N = len(samples)
    rolled_amps = np.roll(samples, -1)
    difsum = sum(np.abs(samples[:-1] - rolled_amps[:-1]))
    del rolled_amps
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(samples))) * 100

# Shimmer compared to n (count) neighboring samples
def get_shim_apqx(samples: np.ndarray, sample_rate: int | float, count: int) -> float:
    N = len(samples)
    m = int((count - 1) / 2)
    difsum = 0
    for i in range(m, N - m):
        difsum += np.abs(samples[i] - np.average(samples[i-m:i+m+1]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(samples))) * 100

# TODO: Further harmonic features from Chaiwongyen et al, 2023 / Li et al, 2022

# Harmonic Noise Ratio (HNR)
# log of harmonic power divided by the residual of subtracting harmonic power from the total power (noise)
# per Chaiwongyen et al, 2022/2023 and Li et al, 2022
def get_hnrs(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    harmonics, magnitudes = lr.core.piptrack(y=samples, sr=sample_rate, fmin=universal_fmin, fmax=universal_fmax)
    harmonic_powers = np.sum(magnitudes, axis=0)
    power_totals = np.sum(gen_stft_mags(samples, sample_rate), axis=0)
    # prevent div0
    harmonic_powers_temp = harmonic_powers[power_totals - harmonic_powers != 0]
    power_totals = power_totals[power_totals - harmonic_powers != 0]
    harmonic_powers = harmonic_powers_temp
    del harmonic_powers_temp
    # prevent log(0)
    harm_ratio = harmonic_powers/(power_totals-harmonic_powers)
    harm_ratio = harm_ratio[harm_ratio != 0]
    local_hnrs = 20 * np.log10(harm_ratio)
    return local_hnrs

# Onset strength (per Li et al, 2022)
def get_onset_strength(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    return lr.onset.onset_strength_multi(y=samples, sr=sample_rate, hop_length=160)

# Intensity (per Li et al, 2022)
def get_intensity(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    stft = gen_stft_mags(samples, sample_rate)
    amps = np.sum(stft, axis=1)
    return lr.power_to_db(amps)

# Chromagram (per Li et al, 2022)
def gen_chromagram(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    return lr.feature.chroma_stft(y=samples, sr=sample_rate, n_fft=12, hop_length=160)

# Estimation of socio-linguistic features from Khanjani et al, 2023

# Estimate pitches based on the maximum power harmonics
def get_pitches(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    harmonics, magnitudes = lr.core.piptrack(y=samples, sr=sample_rate, fmin=universal_fmin, fmax=universal_fmax)
    mag_len = magnitudes.shape[1]
    max_indices = np.argmax(magnitudes, axis=0)
    del magnitudes
    pitches = np.ndarray([])
    pitches = harmonics[max_indices, range(mag_len)]
    return pitches

# Estimate the fluctuations in pitches based on a given offset
# Example, with an offset of 1, every pitch index i will be compared to the pitch at i-1
def get_pitch_fluctuation(samples: np.ndarray, sample_rate: int | float, compare_offset: int) -> np.ndarray:
    if compare_offset < 0:
        raise Exception("compare_offset must be positive")
    pitches = get_pitches(samples, sample_rate)
    comp_pitches = np.roll(pitches, compare_offset)
    fluctuations = pitches - comp_pitches
    del pitches, comp_pitches
    return fluctuations[compare_offset:]

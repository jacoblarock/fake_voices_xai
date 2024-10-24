from typing import Tuple, Union
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt

universal_fmin = 65
universal_fmax = 2093

# File loaders
def lr_load_file(filepath: str) -> Tuple[np.ndarray, Union[int, float]]:
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

# Summarize the results of a calculation resulting in an array into quartiles
# How does this affect classification?
def get_summary_stats(data: np.ndarray) -> dict:
    data = data[data != 0.0]
    return {
        0.05: np.quantile(data, 0.05),
        0.25: np.quantile(data, 0.25),
        0.50: np.quantile(data, 0.50),
        0.75: np.quantile(data, 0.75),
        0.95: np.quantile(data, 0.95)
    }

def gen_mel_spec_lr(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
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

def gen_mfcc_lr(samples: np.ndarray, sample_rate: int | float, count: int) -> np.ndarray:
    mel_spec = gen_mel_spec_lr(samples, sample_rate)
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

# TODO: Harmonic features from Chaiwongyen et al, 2023 / Li et al, 2022

# Harmonic Noise Ratio (HNR)
# log of harmonic power divided by the residual of subtracting harmonic power from the total power (noise)
# per Chaiwongyen et al, 2022/2023 and Li et al, 2022
def get_hnrs(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    harmonics, magnitudes = lr.core.piptrack(y=samples, sr=sample_rate, fmin=universal_fmin, fmax=universal_fmax)
    harmonic_powers = np.sum(magnitudes, axis=1)
    power_totals = np.sum(gen_stft_mags(samples, sample_rate), axis=1)
    # prevent div0
    harmonic_powers = harmonic_powers[power_totals - harmonic_powers != 0]
    power_totals = power_totals[power_totals - harmonic_powers != 0]
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
    mag_len = magnitudes.shape[0]
    max_indices = np.argmax(magnitudes, axis=1)
    del magnitudes
    pitches = harmonics[range(mag_len), max_indices]
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

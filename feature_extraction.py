from time import sleep
from typing import Tuple, Union
import librosa as lr
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

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
            sleep(1)
    return samples, sr

def torch_load_file(filepath: str) -> Tuple[torch.Tensor, int]:
    return torchaudio.load(filepath)

def gen_mel_spec_lr(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    # Short-time Fourier transform 
    sgram = lr.stft(samples)
    sgram_mag, sgram_phase = lr.magphase(sgram)
    # return mel_spec powers
    return lr.feature.melspectrogram(S=sgram_mag, sr=sample_rate)

def gen_mfcc_lr(samples: np.ndarray, sample_rate: int | float, count: int) -> np.ndarray:
    mel_spec = gen_mel_spec_lr(samples, sample_rate)
    return lr.feature.mfcc(S=lr.power_to_db(mel_spec), n_mfcc=20)

def gen_mel_spec(samples: torch.Tensor, sample_rate: int) -> torchaudio.transforms.MelSpectrogram:
    transform = torchaudio.transforms.MelSpectrogram(sample_rate)
    return transform(samples)

def gen_mfcc(samples: torch.Tensor, sample_rate: int, count: int):
    transform = torchaudio.transforms.MFCC(sample_rate)
    return transform(samples)

def plot_spec(to_plot: torchaudio.transforms.Spectrogram, title="title", y_label="freq") -> None:
    form_np = to_plot.numpy()
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(lr.power_to_db(form_np[0]), origin="lower", aspect="auto", interpolation="nearest")
    plt.show()
    return

def get_f0_lens(samples: np.ndarray, sample_rate: int | float) -> np.ndarray:
    f0 = lr.yin(samples, fmin=65, fmax=2093)
    return 1 / f0

def get_local_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    rolled_f0 = np.roll(f0_lens, -1)
    difsum = sum(np.abs(f0_lens[:-1] - rolled_f0[:-1]))
    del rolled_f0
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

def get_rap_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    difsum = 0
    for i in range(1, N - 1):
        difsum += np.abs(f0_lens[i] - np.average(f0_lens[i-1:i+2]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

def get_ppq5_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    difsum = 0
    for i in range(2, N - 2):
        difsum += np.abs(f0_lens[i] - np.average(f0_lens[i-2:i+3]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

def get_ppq55_jitter(samples: np.ndarray, sample_rate: int | float) -> float:
    f0_lens = get_f0_lens(samples, sample_rate)
    N = len(f0_lens)
    difsum = 0
    for i in range(27, N - 27):
        difsum += np.abs(f0_lens[i] - np.average(f0_lens[i-27:i+28]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(f0_lens))) * 100

def get_shim_local(samples: np.ndarray, sample_rate: int | float) -> float:
    N = len(samples)
    rolled_amps = np.roll(samples, -1)
    difsum = sum(np.abs(samples[:-1] - rolled_amps[:-1]))
    del rolled_amps
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(samples))) * 100

def get_shim_apqx(samples: np.ndarray, sample_rate: int | float, count: int) -> float:
    N = len(samples)
    m = int((count - 1) / 2)
    difsum = 0
    for i in range(m, N - m):
        difsum += np.abs(samples[i] - np.average(samples[i-m:i+m+1]))
    return ((1 / (N - 1)) * difsum) / ((1 / N) * sum(np.abs(samples))) * 100


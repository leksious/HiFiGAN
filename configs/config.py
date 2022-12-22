import torch
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    mel = 80
    kernel_size = [16, 16, 8]
    kernel_res = [3, 5, 7]
    dilation = [[1, 2], [2, 6], [3, 12]]
    pre_channels = 256



@dataclass
class MPDConfig:
    periods = [2, 3, 5, 7, 11]
    kernel_size = 5
    stride = 3


@dataclass
class MSDConfig:
    kernel_size = 41


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    f_max_loss: int = None

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    # in case when `min` in clamp is 1e-5
    pad_value: float = -11.5129251


@dataclass
class TrainerConfig:
    grad_norm_clip = 10
    num_epoch = 90
    path_to_save = 'saves'
    path_to_data = './data'
    seed = 4
    batch_size = 16
    learning_rate = 2e-4
    # chckp = 'saves/best.pt'
    chckp = None



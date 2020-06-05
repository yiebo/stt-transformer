import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torchaudio
from torchaudio import transforms

from text.cleaners import english_cleaners as clean_text
from text.symbols import symbols

import numpy as np
from tqdm import tqdm

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
mel_min_max = np.log([2.6e-5, 3000.])
mel_mean = np.mean(mel_min_max)
mel_range = mel_min_max[1] - np.mean(mel_min_max)
sample_rate = 16000

def preprocess(file_path='../DATASETS/LJSpeech-1.1/metadata.csv', root_dir='../DATASETS/LJSpeech-1.1'):
    with open(file_path, encoding='utf8') as file:
        data_ = [line.strip().split('|') for line in file]
    root_dir = root_dir
    sample_rate = 8000
    resample = transforms.Resample(orig_freq=22050, new_freq=sample_rate)
    spectogram = transforms.Spectrogram(n_fft=1024, hop_length=256)
    to_mel = transforms. MelScale(n_mels=80, sample_rate=sample_rate,
                                  n_stft=1024 // 2 + 1)

    mel_data = torch.zeros(len(data_), 316, 80)
    mel_len = torch.empty(len(data_), dtype=torch.int)

    for idx, data in enumerate(tqdm(data_)):
        path, text = data[0], data[1]
        path = f'{root_dir}/wavs/{path}.wav'

        data, sample_rate = torchaudio.load(path)
        data = resample(data)
        data = spectogram(data)
        data = to_mel(data)
        data = data.transpose(1, 2).squeeze(0)
        mel_data[idx, :data.size(0)] = data
        mel_len[idx] = data.size(0)

    torch.save(mel_data, f'{root_dir}/mel_data.pt')
    torch.save(mel_len, f'{root_dir}/mel_len.pt')


def scale_mel(mel, scale=10.):
    mel = torch.log(mel)
    mel = ((mel - mel_mean) / mel_range).clamp(-1, 1)
    # mel_data = mel_data.clamp(mel_min_max[0], mel_min_max[1])
    mel = scale * mel + scale
    return mel

def rescale_mel(mel, scale=10.):
    mel = (mel - scale) / scale
    mel = ((mel * mel_range) + mel_mean)
    # mel_data = mel_data.clamp(mel_min_max[0], mel_min_max[1])
    mel = torch.exp(mel).clamp(mel_min_max[0], mel_min_max[1])
    return mel

class MelWav(nn.Module):
    def __init__(self, n_fft=1024, n_mels=80):
        super().__init__()
        self.mel_to_lin = transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels,
                                                     sample_rate=sample_rate, max_iter=2048)
        self.griffin_lim = transforms.GriffinLim(n_fft=n_fft, hop_length=256)

    def forward(self, x):
        """
        x: [T, C]
        """
        x = x.transpose(0, 1)
        x = rescale_mel(x)
        x = self.mel_to_lin(x)
        x = self.griffin_lim(x)
        return x

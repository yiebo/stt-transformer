from tqdm import tqdm
import glob
import os
import numpy as np
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torchvision import utils
import torchaudio
from torchaudio.transforms import GriffinLim, InverseMelScale, Resample, Spectrogram, MelScale
from ops import positional_encoding

from util import to_device, plot_att_heads, text_id_to_string
from model import Encoder, Decoder
from dataset import Dataset, _symbol_to_id
from audio_process import sample_rate, rescale_mel, scale_mel, MelWav
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_total = 64
batch_size = 4
enc_lr = 0.0001
dec_lr = 0.0005
emb_lr = 0.0001
sym_dim = len(_symbol_to_id)
mel_lin = InverseMelScale(n_stft=1024 // 2 + 1, n_mels=80, sample_rate=sample_rate,
                          max_iter=2*2048).to(device)
griffin_lim = GriffinLim(n_fft=1024, hop_length=256).to(device)

writer = tensorboard.SummaryWriter(log_dir=f'logs/test')

dataset = Dataset('../DATASETS/LJSpeech-1.1/metadata.csv', '../DATASETS/LJSpeech-1.1')
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=batch_size,
                        shuffle=False, num_workers=0, drop_last=True)

resample = Resample(orig_freq=22050, new_freq=sample_rate)
spectogram = Spectrogram(n_fft=1024, hop_length=256).to(device)
to_mel = MelScale(n_mels=80, sample_rate=sample_rate,
                  n_stft=1024 // 2 + 1).to(device)
with open('../DATASETS/LJSpeech-1.1/metadata.csv', encoding='utf8') as file:
    data = [line.strip().split('|') for line in file]
path, text = data[0][0], data[0][1]
path = f'../DATASETS/LJSpeech-1.1/wavs/{path}.wav'
data, sr = torchaudio.load(path)

data = resample(data)
data = data.to(device)

data = spectogram(data.squeeze(0))
mel_norm = ((data.unsqueeze(0) - data.mean()) / data.std()).clamp(-1, 1) * .5 + .5
writer.add_image(f'spec/origin', mel_norm, 0)
writer.add_audio(f'audio/origin', griffin_lim(data), global_step=0, sample_rate=sample_rate)

data = to_mel(data)
data = scale_mel(data)

data = rescale_mel(data)
data = mel_lin(data)
mel_norm = ((data.unsqueeze(0) - data.mean()) / data.std()).clamp(-1, 1) * .5 + .5

writer.add_image(f'spec/re', mel_norm, 0)
writer.add_audio(f'audio/re', griffin_lim(data), global_step=0, sample_rate=sample_rate)

mel_wav = MelWav().to(device)

for batch in dataloader:
    text_data, text_pos, text_len, text_mask, mel_data, mel_pos, mel_len, mel_mask, gate = to_device(batch, device)
    start = time.time()
    # data = mel_wav(mel_data, mel_mask[:, -1].unsqueeze(1))
    
    x = mel_data.transpose(-2, -1)
    x = rescale_mel(x)
    x = mel_lin(x)
    mel_norm = ((x - x.mean()) / x.std()).clamp(-1, 1) * .5 + .5
    writer.add_image(f'spec/all2', mel_norm[:1], 0)
    x = griffin_lim(x)
    for sample in x:
        writer.add_audio(f'audio/all2', sample, global_step=0, sample_rate=sample_rate)
    print(time.time() - start)

    start = time.time()
    for data, mel_len_ in zip(mel_data, mel_len):
        writer.add_audio(f'audio/all', mel_wav(data[:mel_len_]), global_step=0, sample_rate=sample_rate)
    print(time.time() - start)
    
    writer.flush()
    exit()

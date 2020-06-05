import torch
from torch.nn.utils import rnn
from torch.utils import data
import torchaudio
from torchaudio import transforms

from text.cleaners import english_cleaners as clean_text
from text.symbols import _symbol_to_id, _id_to_symbol
from audio_process import scale_mel, sample_rate

import numpy as np

def parse_text(text):
    text = clean_text(text)
    text_sequence = []
    for s in text:
        if s in _symbol_to_id:
            text_sequence.append(_symbol_to_id[s])

    return torch.tensor(text_sequence)


class Dataset(data.Dataset):
    def __init__(self, file_path, root_dir, mel_scale=1):
        with open(file_path, encoding='utf8') as file:
            self.data = [line.strip().split('|') for line in file]
        self.root_dir = root_dir
        self.mel_scale = mel_scale
        self.mel_data_padded = torch.load(f'{root_dir}/mel_data_{sample_rate}.pt')
        self.mel_data_len = torch.load(f'{root_dir}/mel_len_{sample_rate}.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        mel_data: [T, C]
        text_data: [T]
        """
        path, text = self.data[idx][0], self.data[idx][1]
        path = f'{self.root_dir}/wavs/{path}.wav'
        text_data = parse_text(text)

        mel_len = self.mel_data_len[idx]

        mel_data = self.mel_data_padded[idx, :mel_len]
        mel_data = scale_mel(mel_data)
        # mel_data = mel_data.clamp(mel_min_max[0], mel_min_max[1])
        return text_data, mel_data

    def collocate(self, batch):
        """
        batch: B * [text_data: [T], mel_data: [T, C]]
        -----
        return: text_data, text_len, text_mask, mel_data, mel_len, mel_mask
            text_data: [B, T], text_len: [B], text_mask: [B, 1, T]
            mel_data: [B, T, C], mel_len: [B], mel_mask: [B, T, T]
            gate: [B, T, 1]
        """
        # sort on text size
        batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        text_data, mel_data = zip(*batch)

        text_max_len = text_data[0].size(0)
        mel_max_len = max([mel.size(0) for mel in mel_data])

        text_pad = torch.zeros(len(text_data), text_max_len, dtype=text_data[0].dtype)
        for idx, text in enumerate(text_data):
            size = torch.randint(5, 10, [1])
            pad = text[-size:]
            size = (text_max_len / size) + 1
            text_pad[idx] = pad.repeat(size)[:text_max_len]
            text_pad[idx, :text.size(0)] = text

        text_data = rnn.pack_sequence(text_data)
        text_data, text_len = rnn.pad_packed_sequence(text_data, batch_first=True, padding_value=0)
        text_pos = torch.arange(0, text_max_len).view(1, -1) + 1

        mel_data = rnn.pack_sequence(mel_data, enforce_sorted=False)
        mel_data, mel_len = rnn.pad_packed_sequence(mel_data, batch_first=True,
                                                    padding_value=0, total_length=mel_max_len)
        mel_pos = torch.arange(0, mel_max_len).view(1, -1) + 1
        # -----
        text_mask = (text_pos > text_len.unsqueeze(1)).unsqueeze(1)
        mel_mask = (mel_pos > mel_len.unsqueeze(1)).unsqueeze(1)

        # mask_noise = torch.arange(mel_mask.size(-1)).unsqueeze(0) - (mel_len - 1).unsqueeze(-1)
        # mask_noise = torch.sigmoid(mask_noise.to(torch.float)) * 0.0999 + 0.0001
        # mask_noise = torch.randn_like(mask_noise) * mask_noise
        # mel_mask = mel_mask + mask_noise.unsqueeze(1)
        # mel_mask = mel_mask.clamp(0, 1)

        gate = torch.arange(text_len[0]).unsqueeze(0) >= (text_len - 1).unsqueeze(-1)
        gate = gate.unsqueeze(-1)

        text_att_mask = torch.triu(torch.ones(text_mask.size(2), text_mask.size(2), dtype=torch.bool), 1)
        text_mask = text_att_mask.unsqueeze(0)

        text_mask = text_mask.to(torch.float)
        mel_mask = mel_mask.to(torch.float)
        gate = gate.to(torch.float)

        return text_data, text_pos, text_len, text_mask, mel_data, mel_pos, mel_len, mel_mask, gate, text_pad

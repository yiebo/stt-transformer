import torch
from torch.nn.utils import rnn
from torch.utils import data
import torchaudio
from torchaudio import transforms

from text.cleaners import english_cleaners as clean_text
from text.symbols import symbols
from audio_process import scale_mel

import numpy as np

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

class Dataset(data.Dataset):
    def __init__(self, file_path, root_dir):
        with open(file_path, encoding='utf8') as file:
            self.data = [line.strip().split('|') for line in file]
        self.root_dir = root_dir

        self.text_pad = _symbol_to_id['_']
        self.mel_data_padded = torch.load(f'{root_dir}/mel_data.pt')
        self.mel_data_len = torch.load(f'{root_dir}/mel_len.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        mel_data: [T, C]
        text_data: [T]
        """
        path, text = self.data[idx][0], self.data[idx][1]
        path = f'{self.root_dir}/wavs/{path}.wav'
        text = clean_text(text)
        text_sequence = []
        for s in text:
            if s in _symbol_to_id:
                text_sequence.append(_symbol_to_id[s])

        text_sequence = torch.tensor(text_sequence)
        mel_len = self.mel_data_len[idx]

        mel_data = self.mel_data_padded[idx, :mel_len]
        mel_data = scale_mel(mel_data)
        return text_sequence, mel_data

    def collocate(self, batch):
        """
        batch: B * [text_data: [T], mel_data: [T, C]]
        -----
        return: text_data, text_len, text_mask, mel_data, mel_len, mel_mask
            text_data: [B, T], text_len: [B, T], text_mask: [B, 1, T]
            mel_data: [B, T, C], mel_len: [B, T], mel_mask: [B, T, T]
            gate: [B, T, 1]
        """

        batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)

        text_data, mel_data = [], []
        text_pos, mel_pos = [], []
        for text, mel in batch:
            mel_data.append(mel)
            mel_pos.append(torch.arange(mel.size(0)) + 1)
            text_data.append(text)
            text_pos.append(torch.arange(text.size(0)) + 1)

        text_data = rnn.pack_sequence(text_data)
        text_data, text_len = rnn.pad_packed_sequence(text_data, batch_first=True, padding_value=self.text_pad)

        # pad so it is scalable
        mel_max_len = max([mel.size(0) for mel in mel_data])

        mel_data = rnn.pack_sequence(mel_data, enforce_sorted=False)
        mel_data, mel_len = rnn.pad_packed_sequence(mel_data, batch_first=True,
                                                    padding_value=0, total_length=mel_max_len)
        # -----
        text_pos = rnn.pack_sequence(text_pos)
        text_pos, text_len = rnn.pad_packed_sequence(text_pos, batch_first=True, padding_value=0)

        mel_pos = rnn.pack_sequence(mel_pos, enforce_sorted=False)
        mel_pos, mel_len = rnn.pad_packed_sequence(mel_pos, batch_first=True,
                                                   padding_value=0, total_length=mel_max_len)

        text_mask = (text_pos == 0).unsqueeze(1)
        mel_mask = (mel_pos == 0).unsqueeze(1)

        gate = torch.arange(text_len[0]).unsqueeze(0) >= (text_len - 1).unsqueeze(-1)
        gate = gate.unsqueeze(-1).to(torch.float)

        text_att_mask = torch.triu(torch.ones(text_mask.size(2), text_mask.size(2), dtype=torch.bool), 1)
        text_mask = text_att_mask + text_mask

        return text_data, text_pos, text_len, text_mask, mel_data, mel_pos, mel_len, mel_mask, gate

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
from ops import positional_encoding

from util import to_device, plot_att_heads
from model import Encoder, Decoder
from dataset import Dataset, _symbol_to_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_total = 64
batch_size = 16
enc_lr = 0.0001
dec_lr = 0.0005
emb_lr = 0.0001

# -----------------------------------

text_embedding = nn.Embedding(num_embeddings=len(_symbol_to_id), embedding_dim=512).to(device)
pos_embedding = nn.Embedding.from_pretrained(positional_encoding(512, 512), freeze=True).to(device)
pos_embedding_ = nn.Embedding.from_pretrained(positional_encoding(256, 512), freeze=True).to(device)

encoder = Encoder(emb_channels=512).to(device)
decoder = Decoder(mel_channels=80, enc_channels=512, emb_channels=512).to(device)

optimizer = torch.optim.Adam([{'params': text_embedding.parameters(), 'lr': emb_lr},
                              {'params': encoder.parameters(), 'lr': enc_lr},
                              {'params': decoder.parameters(), 'lr': dec_lr}],
                             lr=0.001)

# -----------------------------------

logs_idx = f'emb_lr{emb_lr}-enc_lr{enc_lr}-dec_lr{dec_lr}-batch_size{batch_size}2'
saves = glob.glob(f'logs/{logs_idx}/*.pt')
dataset = Dataset('../DATASETS/LJSpeech-1.1/metadata.csv', '../DATASETS/LJSpeech-1.1/wavs')
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=1,
                        shuffle=True, num_workers=0, drop_last=True)
writer = tensorboard.SummaryWriter(log_dir=f'test/{logs_idx}')
if len(saves) != 0:
    saves.sort(key=os.path.getmtime)
    checkpoint = torch.load(saves[-1], )
    text_embedding.load_state_dict(checkpoint['text_embedding'])
    text_embedding.eval()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()

for idx, batch in enumerate(tqdm(BackgroundGenerator(dataloader), total=len(dataloader))):
    text_data, text_pos, text_mask, mel_data, mel_pos, mel_mask, gate = to_device(batch, device)
    # audio_data = F.avg_pool1d(audio_data, kernel_size=2, padding=1)
    with torch.no_grad():
        text_emb = text_embedding(text_data)
        text_pos_emb = pos_embedding_(text_pos)
        enc_out, att_heads_enc = encoder(text_emb, text_mask, text_pos_emb)

        mel_pos = torch.arange(1, 512).view(1, 511).to(device)
        mel_pos_emb_ = pos_embedding(mel_pos)
        mel_mask_ = torch.triu(torch.ones(511, 511, dtype=torch.bool), 1).unsqueeze(0).to(device)
        # [B, T, C], [B, T, C], [B, T, 1], [B, T, T_text]
        mel = torch.zeros(1, 511, 80).to(device)
        for pos_idx in tqdm(range(511)):
            mel_pos_emb = mel_pos_emb_[:, :pos_idx + 1]
            mel_mask = mel_mask_[:, :pos_idx + 1, :pos_idx + 1]
            mels_out, mels_out_post, gates_out, att_heads_dec, att_heads = decoder(mel[:, :pos_idx + 1], enc_out,
                                                                                   mel_mask, text_mask, mel_pos_emb)

            mel[:, pos_idx] = mels_out_post[:, pos_idx]
            if gates_out[0, -1, 0] > .9:
                break
        print(gates_out)

    # [B, 1, T, 1], [B, 1, T, C]
    mel_data = mel_data.unsqueeze(1).transpose(2, 3)
    gate = gate.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
    gate = utils.make_grid(gate, nrow=1, padding=2, pad_value=1, normalize=True, range=(0, 1))
    mel_data = utils.make_grid(mel_data, nrow=1, padding=2, pad_value=1, normalize=True, scale_each=True)
    mel_data = torch.cat([mel_data, gate], 1)
    # writer.add_image(f'mel/target', mel_data, global_idx)

    mel = mel.unsqueeze(1).transpose(2, 3)
    gates_out = gates_out.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
    gates_out = utils.make_grid(gates_out, nrow=1, padding=2, pad_value=1, normalize=True, range=(0, 1))
    mel = utils.make_grid(mel, nrow=1, padding=2, pad_value=1,
                          normalize=True, scale_each=True)
    mel = torch.cat([mel, gates_out], 1)
    # writer.add_image(f'mel/post_prediction', mels_out_post, global_idx)
    writer.add_image(f'mel', torch.cat([mel_data, mel], 2), idx)

    writer.add_image(f'attention', plot_att_heads(att_heads, 0), idx)
    writer.add_image(f'attention/enc', plot_att_heads(att_heads_enc, 0), idx)
    writer.add_image(f'attention/dec', plot_att_heads(att_heads_dec, 0), idx)
    exit()

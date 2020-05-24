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
from torchaudio.transforms import GriffinLim, InverseMelScale
from ops import positional_encoding

from util import to_device, plot_att_heads, text_id_to_string
from model import Encoder, Decoder
from dataset import Dataset, _symbol_to_id, sample_rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_total = 64
batch_size = 16
enc_lr = 0.0001
dec_lr = 0.0005
emb_lr = 0.0001
sym_dim = len(_symbol_to_id)

griffin_lim = GriffinLim(n_fft=1024, hop_length=256).to(device)
mel_lin = InverseMelScale(n_stft=1024, n_mels=80, sample_rate=sample_rate, f_min=1e-5, f_max=3000.).to(device)

# -----------------------------------

text_embedding = nn.Embedding(num_embeddings=sym_dim, embedding_dim=256).to(device)
pos_embedding_mel = nn.Embedding.from_pretrained(positional_encoding(512, 256), freeze=True).to(device)
pos_embedding_text = nn.Embedding.from_pretrained(positional_encoding(256, 256), freeze=True).to(device)

encoder = Encoder(in_channels=80, emb_channels=256).to(device)
decoder = Decoder(in_channels=256, out_channels=sym_dim, enc_channels=256, emb_channels=256).to(device)

optimizer = torch.optim.Adam([{'params': text_embedding.parameters(), 'lr': emb_lr},
                              {'params': encoder.parameters(), 'lr': enc_lr},
                              {'params': decoder.parameters(), 'lr': dec_lr}],
                             lr=0.001)

# -----------------------------------

logs_idx = f'emb_lr{emb_lr}-enc_lr{enc_lr}-dec_lr{dec_lr}-batch_size{batch_size}_'
saves = glob.glob(f'logs/{logs_idx}/*.pt')
dataset = Dataset('../DATASETS/LJSpeech-1.1/metadata.csv', '../DATASETS/LJSpeech-1.1')
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)
writer = tensorboard.SummaryWriter(log_dir=f'logs/{logs_idx}')
if len(saves) != 0:
    saves.sort(key=os.path.getmtime)
    checkpoint = torch.load(saves[-1], )
    text_embedding.load_state_dict(checkpoint['text_embedding'])
    text_embedding.train()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.train()
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.train()
    optimizer.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint['epoch']
    global_idx = checkpoint['global_idx']
else:
    epoch = 0
    global_idx = 0


# ---------------------------------------

summ_counter = 0
mean_losses = np.zeros(3)
mean_metrics = np.zeros(4)
for epoch in tqdm(range(epoch, epoch_total),
                  initial=epoch, total=epoch_total, leave=False, dynamic_ncols=True):
    for idx, batch in enumerate(tqdm(BackgroundGenerator(dataloader),
                                     total=len(dataloader), leave=False, dynamic_ncols=True)):
        text_data, text_pos, text_len, text_mask, mel_data, mel_pos, mel_len, mel_mask, gate = to_device(batch, device)

        mel_pos_emb = pos_embedding_mel(mel_pos)
        enc_out, att_heads_enc = encoder(mel_data, mel_mask, mel_pos_emb)

        # [B, T, C], [B, T, C], [B, T, 1], [B, T, T_text]
        text_emb = text_embedding(text_data)
        text_pos_emb = pos_embedding_text(text_pos)
        text_out, gate_out, att_heads_dec, att_heads = decoder(text_emb, enc_out, text_mask, mel_mask, text_pos_emb)

        loss_text = F.cross_entropy(text_out.view(-1, sym_dim), text_data.view(-1), ignore_index=0)
        loss_gate = F.binary_cross_entropy(gate_out, gate)
        loss = loss_text + loss_gate

        optimizer.zero_grad()
        loss.backward()

        grad_norm_enc = nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        grad_norm_dec = nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        optimizer.step()

        # -----------------------------------------

        global_idx += 1
        summ_counter += 1
        mean_losses += [loss_text.item(),
                        loss_gate.item(),
                        loss.item()]
        mean_metrics += [grad_norm_enc,
                         grad_norm_dec,
                         encoder.pos_alpha.item(),
                         decoder.pos_alpha.item()]

        if global_idx % 10 == 0:
            # print(attentions_t[:4])
            mean_losses /= summ_counter
            mean_metrics /= summ_counter
            writer.add_scalar('loss/text', mean_losses[0], global_idx)
            writer.add_scalar('loss/gate', mean_losses[1], global_idx)
            writer.add_scalar('loss_total', mean_losses[2], global_idx)

            writer.add_scalar('grad_norm/enc', mean_metrics[0], global_idx)
            writer.add_scalar('grad_norm/dec', mean_metrics[1], global_idx)
            writer.add_scalar('alpha/enc', mean_metrics[2], global_idx)
            writer.add_scalar('alpha/dec', mean_metrics[3], global_idx)
            mean_losses = np.zeros(3)
            mean_metrics = np.zeros(4)
            summ_counter = 0

            for mel_data_, mel_len_ in zip(mel_data[:4], mel_len[:4]):
                mel_data_ = mel_data_[:mel_len_].transpose(0, 1)
                print(mel_data_.size())
                mel_min_max = np.log([1e-5, 3000.])
                mel_range = mel_min_max[1] - np.mean(mel_min_max)
                mel_mean = np.mean(mel_min_max)
                mel_data_ = (mel_data_ - 10) / 10.
                mel_data_ = mel_data_ * mel_range + mel_mean
                mel_data_ = mel_data_.exp()
                print(mel_data_.size())
                mel_data_ = mel_lin(mel_data_)
                mel_data_ = griffin_lim(mel_data_)
                writer.add_audio(f'audio', mel_data_, global_step=global_idx, sample_rate=sample_rate)

            _, text_out = text_out.max(-1)
            for text_idx, text in enumerate(text_id_to_string(text_data[:4], text_len[:4])):
                writer.add_text(f'text/input', text, global_idx)

            for text_idx, text in enumerate(text_id_to_string(text_out[:4], text_len[:4])):
                writer.add_text(f'text/output', text, global_idx)

            # [B, 1, T, 1], [B, 1, T, C]
            mel_data = mel_data.unsqueeze(1).transpose(2, 3)
            mel_data = utils.make_grid(mel_data[:4], nrow=1, padding=2, pad_value=1, normalize=True, scale_each=True)
            writer.add_image(f'mel', mel_data, global_idx)

            gate = gate.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
            gate_out = gate_out.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
            gate_out = torch.cat([gate, gate_out], 2)
            gate_out = utils.make_grid(gate_out[:4], nrow=1, padding=2, pad_value=.5,
                                       normalize=True, range=(0, 1))
            writer.add_image(f'gate', gate_out, global_idx)

            writer.add_image(f'attention', plot_att_heads(att_heads, 1), global_idx)
            writer.add_image(f'attention_enc', plot_att_heads(att_heads_enc, 1), global_idx)
            writer.add_image(f'attention_dec', plot_att_heads(att_heads_dec, 1), global_idx)

            if global_idx % 1000 == 0:
                with torch.no_grad():
                    text_pos = torch.arange(1, 256).view(1, 255).expand(4, -1).to(device)
                    text_pos_emb = pos_embedding_text(text_pos)
                    text_mask = torch.triu(torch.ones(256, 256, dtype=torch.bool), 1).unsqueeze(0).to(device)
                    text_data = torch.zeros(4, 255, dtype=text_data.dtype).to(device)
                    enc_out = enc_out[:4]
                    mel_mask = mel_mask[:4]
                    for pos_idx in tqdm(range(255), leave=False, dynamic_ncols=True):
                        text_emb = text_embedding(text_data)
                        (text_out, gate_out,
                         att_heads_dec, att1_heads) = decoder(text_emb[:, :pos_idx + 1], enc_out,
                                                              text_mask[:, :pos_idx + 1, :pos_idx + 1],
                                                              mel_mask, text_pos_emb[:, :pos_idx + 1])

                        text_out = text_out.argmax(-1)
                        text_data[:, pos_idx] = text_out[:, pos_idx]
                    text_len = gate_out.argmax(1).unsqueeze(-1) + 1

                for text_idx, text in enumerate(text_id_to_string(text_out)):
                    writer.add_text(f'text/pred', text, global_idx)

                for text_idx, text in enumerate(text_id_to_string(text_out, text_len)):
                    writer.add_text(f'text/pred2', text, global_idx)

                gate = F.pad(gate, (0, 255 - gate.size(-1)))
                gate_out = gate_out.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
                gate_out = torch.cat([gate[:4], gate_out], 2)
                gate_out = utils.make_grid(gate_out[:4], nrow=1, padding=2, pad_value=.5,
                                           normalize=True, range=(0, 1))
                writer.add_image(f'gate/test', gate_out, global_idx)

                writer.add_image(f'attention/test', plot_att_heads(att_heads, 1), global_idx)
                writer.add_image(f'attention_dec/test', plot_att_heads(att_heads_dec, 1), global_idx)

    saves = glob.glob(f'logs/{logs_idx}/*.pt')
    if len(saves) == 10:
        saves.sort(key=os.path.getmtime)
        os.remove(saves[0])

    torch.save({
        'epoch': epoch + 1,
        'global_idx': global_idx,
        'text_embedding': text_embedding.state_dict(),
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict()},
        f'logs/{logs_idx}/model_{epoch + 1}.pt')

    # check for early exit
    with open('run.txt', 'r+') as run:
        if not int(run.read()):
            run.seek(0)
            run.write('1')
            exit()

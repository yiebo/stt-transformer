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

from util import to_device, plot_att_heads, text_id_to_string, confusion_matrix, plot_heatmap
from model import SST
from dataset import Dataset, _symbol_to_id, sample_rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_total = 64
batch_size = 16
enc_lr = 0.0001
dec_lr = 0.0005

# -----------------------------------
nr_symbols = len(_symbol_to_id)
weights = (torch.arange(nr_symbols) >= 10).to(torch.float)
weights = ((weights + 1) / 2).to(device)
model = SST(mel_channels=80, text_channels=nr_symbols, emb_channels=256).to(device)

optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': enc_lr},
                              {'params': model.decoder.parameters(), 'lr': dec_lr}],
                             lr=0.001)

# -----------------------------------

logs_idx = f'enc_lr{enc_lr}-dec_lr{dec_lr}-batch_size{batch_size}_pos2'
saves = glob.glob(f'logs/{logs_idx}/*.pt')
dataset = Dataset('../DATASETS/LJSpeech-1.1/metadata.csv', '../DATASETS/LJSpeech-1.1')
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)
writer = tensorboard.SummaryWriter(log_dir=f'logs/{logs_idx}')
if len(saves) != 0:
    saves.sort(key=os.path.getmtime)
    checkpoint = torch.load(saves[-1], )
    model.load_state_dict(checkpoint['model'])
    model.train()
    optimizer.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint['epoch']
    global_idx = checkpoint['global_idx']
else:
    epoch = 0
    global_idx = 0

# ---------------------------------------

summ_counter = 0
mean_losses = np.zeros(3)
mean_metrics = np.zeros(5)
for epoch in tqdm(range(epoch, epoch_total),
                  initial=epoch, total=epoch_total, leave=False, dynamic_ncols=True):
    for idx, batch in enumerate(tqdm(BackgroundGenerator(dataloader),
                                     total=len(dataloader), leave=False, dynamic_ncols=True)):
        text_data, text_pos, text_len, text_mask, mel_data, mel_pos, mel_len, mel_mask, gate, text_data_ = to_device(batch, device)

        text_out, gate_out, att_heads_enc, att_heads_dec, att_heads = model(mel_data, mel_pos, mel_mask,
                                                                            text_data_, text_pos, text_mask)

        loss_text = F.cross_entropy(text_out.view(-1, nr_symbols), text_data.view(-1), weight=weights, ignore_index=0)
        loss_gate = F.binary_cross_entropy(gate_out, gate)
        loss = loss_text + loss_gate
        optimizer.zero_grad()
        loss.backward()
        # print(mel_mask.grad)

        grad_norm_enc = nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        grad_norm_dec = nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)

        optimizer.step()

        # -----------------------------------------

        gate_val = torch.gather(gate_out.squeeze(-1), 1, text_len.unsqueeze(-1) - 1).mean()
        global_idx += 1
        summ_counter += 1
        mean_losses += [loss_text.item(),
                        loss_gate.item(),
                        loss.item()]
        mean_metrics += [grad_norm_enc,
                         grad_norm_dec,
                         model.encoder.pos_alpha.item(),
                         model.decoder.pos_alpha.item(),
                         gate_val.item()]

        if global_idx % 100 == 0:
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
            writer.add_scalar('gate_val', mean_metrics[4], global_idx)
            mean_losses = np.zeros_like(mean_losses)
            mean_metrics = np.zeros_like(mean_metrics)
            summ_counter = 0

            writer.add_figure('batch_CM', confusion_matrix(text_out, text_data, [*_symbol_to_id]), global_idx)

            writer.add_text(f'text/input', text_id_to_string(text_data[:4], text_len=text_len[:4]), global_idx)

            gate_out[:, -1] = 1
            text_len = (gate_out.squeeze(-1) > .9).to(torch.float)
            text_len = text_len / (torch.arange(gate_out.size(1), device=device) + 1.)
            text_len = text_len.argmax(-1) + 1

            text_out = text_out.argmax(-1)
            writer.add_text(f'text/output', text_id_to_string(text_out[:4], text_len=text_len[:4]), global_idx)

            # [B, 1, T, 1], [B, 1, T, C]
            # mel_data = mel_data.unsqueeze(1).transpose(2, 3)
            # mel_data = utils.make_grid(mel_data[:4], nrow=1, padding=2, pad_value=1, normalize=True, scale_each=True)
            # writer.add_image(f'mel', mel_data, global_idx)

            gate = gate.expand(-1, -1, 15).transpose(1, 2)
            gate_out = gate_out.expand(-1, -1, 15).transpose(1, 2)
            gate_out = torch.cat([gate, gate_out], 1)

            writer.add_figure('enc_mask', plot_heatmap(mel_mask[:4].expand(-1, 30, -1)), global_idx)
            writer.add_figure('gate_heatmap', plot_heatmap(gate_out[:4]), global_idx)

            writer.add_image(f'attention', plot_att_heads(att_heads, 1), global_idx)
            writer.add_image(f'attention_enc', plot_att_heads(att_heads_enc, 1), global_idx)
            writer.add_image(f'attention_dec', plot_att_heads(att_heads_dec, 1), global_idx)

            if global_idx % 1000 == 0:
                with torch.no_grad():
                    model.eval()
                    text_out, gate_out, att_heads_enc, att_heads_dec, att_heads = model(mel_data, mel_pos, mel_mask)
                    model.train()
                gate_out[:, -1] = 1
                text_len = (gate_out.squeeze(-1) > .9).to(torch.float)
                text_len = text_len / (torch.arange(gate_out.size(1), device=device) + 1.)
                text_len = text_len.argmax(-1) + 1

                text_out = text_out.argmax(-1)
                writer.add_text(f'text/pred_clip', text_id_to_string(text_out, text_len=text_len), global_idx)
                writer.add_text(f'text/pred', text_id_to_string(text_out), global_idx)
                
                gate_out = gate_out.expand(-1, -1, 30).transpose(1, 2)
                writer.add_figure('gate_heatmap/test', plot_heatmap(gate_out[:4]), global_idx)
                writer.add_image(f'attention/test', plot_att_heads(att_heads, 1), global_idx)
                writer.add_image(f'attention_dec/test', plot_att_heads(att_heads_dec, 1), global_idx)

    saves = glob.glob(f'logs/{logs_idx}/*.pt')
    if len(saves) == 10:
        saves.sort(key=os.path.getmtime)
        os.remove(saves[0])

    torch.save({
        'epoch': epoch + 1,
        'global_idx': global_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        f'logs/{logs_idx}/model_{epoch + 1}.pt')

    # check for early exit
    with open('run.txt', 'r+') as run:
        if not int(run.read()):
            run.seek(0)
            run.write('1')
            exit()

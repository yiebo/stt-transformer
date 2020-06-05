import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import EncoderBlock, DecoderBlock
from ops import positional_encoding
from dataset import _symbol_to_id


class PostNet(nn.Module):
    def __init__(self, mel_channels):
        super().__init__()
        conv_layers = []
        mapping = [mel_channels, 256, 256, 256, 256, mel_channels]
        for idx in range(len(mapping) - 1):
            block = nn.Sequential(nn.Conv1d(mapping[idx], mapping[idx + 1], kernel_size=5, padding=0),
                                  nn.BatchNorm1d(mapping[idx + 1]))
            conv_layers.append(block)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        x = x.transpose(1, 2)
        for conv in self.conv_layers[:-1]:
            x = F.pad(x, [4, 0])
            x = conv(x)
            x = F.tanh(x)
            x = F.dropout(x, .1, self.training)
        x = F.pad(x, [4, 0])
        x = self.conv_layers[-1](x)
        x = x.transpose(1, 2)
        return x


class EncoderPre(nn.Module):
    def __init__(self, in_channels, emb_channels):
        super().__init__()
        self.pre_net = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels, emb_channels, kernel_size=5, padding=5 // 2),
                          nn.BatchNorm1d(emb_channels)),
            nn.Sequential(nn.Conv1d(emb_channels, emb_channels, kernel_size=5, padding=5 // 2),
                          nn.BatchNorm1d(emb_channels)),
            nn.Sequential(nn.Conv1d(emb_channels, emb_channels, kernel_size=5, padding=5 // 2),
                          nn.BatchNorm1d(emb_channels)),
        ])
        self.linear = nn.Linear(emb_channels, emb_channels)

    def forward(self, x):
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        for conv in self.pre_net:
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, 0.1, self.training)
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, emb_channels):
        super().__init__()
        self.pre_net = EncoderPre(in_channels, emb_channels)
        self.pos_alpha = nn.Parameter(torch.ones(1))

        self.enc_attention = nn.ModuleList([
            EncoderBlock(emb_channels, out_channels=emb_channels, emb_channels=64, nr_heads=4),
            EncoderBlock(emb_channels, out_channels=emb_channels, emb_channels=64, nr_heads=4),
            EncoderBlock(emb_channels, out_channels=emb_channels, emb_channels=64, nr_heads=4)
        ])

    def forward(self, x, mask, pos_emb):
        # [B, T, C] -> [B, C, T]
        x = self.pre_net(x)
        x = x + self.pos_alpha * pos_emb
        x = F.dropout(x, .1, self.training)

        att_heads = []
        for att_block in self.enc_attention:
            x, att = att_block(x, mask)

            att_heads.append(att)

        return x, tuple(att_heads)


class Decoder(nn.Module):
    def __init__(self, text_channels, enc_channels, emb_channels):
        super().__init__()
        self.text_embedding = nn.Embedding(num_embeddings=text_channels, embedding_dim=emb_channels)
        self.pre_net = nn.ModuleList([
            nn.Linear(emb_channels, emb_channels),
            nn.Linear(emb_channels, emb_channels),
        ])
        self.norm = nn.Linear(emb_channels, emb_channels)
        self.pos_alpha = nn.Parameter(torch.ones(1))

        self.dec_attention = nn.ModuleList([
            DecoderBlock(emb_channels, enc_channels=enc_channels, out_channels=emb_channels,
                         emb_channels=64, nr_heads=4),
            DecoderBlock(emb_channels, enc_channels=enc_channels, out_channels=emb_channels,
                         emb_channels=64, nr_heads=4),
            DecoderBlock(emb_channels, enc_channels=enc_channels, out_channels=emb_channels,
                         emb_channels=64, nr_heads=4)
        ])

        self.to_out = nn.Linear(emb_channels, text_channels)
        self.to_gate = nn.Linear(emb_channels, 1)

    def forward(self, text_data, enc_out, mask, enc_mask, pos_emb):
        x = self.text_embedding(text_data)
        x_ = torch.zeros(x.size(0), 1, 256, device=x.device)
        x = torch.cat([x_, x[:, :-1]], 1)
        for linear in self.pre_net:
            x = linear(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, .5, self.training)
        x = self.norm(x)

        x = x + self.pos_alpha * pos_emb
        x = F.dropout(x, .1, self.training)
        # 3 * [B, N, T, T], 3 * [B, N, T, T_enc]
        # [B, 1, T, 1]
        att_heads_dec, att_heads = [], []
        for att_block in self.dec_attention:
            x, att_dec, att_weight = att_block(x, enc_out, mask, enc_mask)
            x = F.leaky_relu(x)

            att_heads_dec.append(att_dec)
            att_heads.append(att_weight)

        text_out = self.to_out(x)
        gate_out = self.to_gate(x)

        # [B, T, 1]
        # out_mask = mask[:, -1].unsqueeze(-1)
        # text_out = text_out.masked_fill(out_mask, 0)

        gate_out = torch.sigmoid(gate_out)
        # gate_out = gate_out.masked_fill(out_mask, 1)

        return text_out, gate_out, tuple(att_heads_dec), tuple(att_heads)


class SST(nn.Module):
    def __init__(self, mel_channels, text_channels, emb_channels):
        super().__init__()
        self.text_channels = text_channels
        self.pos_embedding_mel = nn.Embedding.from_pretrained(
            positional_encoding(2 * 512, emb_channels), freeze=True)
        self.pos_embedding_text = nn.Embedding.from_pretrained(
            positional_encoding(2 * 256, emb_channels), freeze=True)

        self.encoder = Encoder(in_channels=80, emb_channels=256)
        self.decoder = Decoder(text_channels, enc_channels=256, emb_channels=256)

    def encode(self, mel_data, mel_pos, mel_mask):
        mel_pos_emb = self.pos_embedding_mel(mel_pos)
        enc_out, att_heads_enc = self.encoder(mel_data, mel_mask, mel_pos_emb)
        return enc_out, att_heads_enc

    def decode(self, enc_out, mel_mask, text_data, text_pos, text_mask):
        text_pos_emb = self.pos_embedding_text(text_pos)
        text_out, gate_out, att_heads_dec, att_heads = self.decoder(text_data, enc_out,
                                                                    text_mask, mel_mask, text_pos_emb)
        return text_out, gate_out, att_heads_dec, att_heads

    def forward(self, mel_data, mel_pos, mel_mask, text_data=None, text_pos=None, text_mask=None):
        enc_out, att_heads_enc = self.encode(mel_data, mel_pos, mel_mask)
        if text_data is not None:
            text_out, gate_out, att_heads_dec, att_heads = self.decode(enc_out, mel_mask,
                                                                       text_data, text_pos, text_mask)
        else:
            batch_size = mel_data.size(0)
            text_pos = torch.arange(1, 2 * 256, device=mel_data.device).view(1, -1)
            text_mask = torch.triu(torch.ones(text_pos.size(1), text_pos.size(1),
                                              dtype=torch.bool, device=mel_data.device), 1).unsqueeze(0)
            text_data = torch.zeros(batch_size, text_pos.size(1), dtype=torch.long, device=mel_data.device)

            for pos_idx in range(text_pos.size(1)):
                text_out, gate_out, att_heads_dec, att_heads = self.decode(enc_out, mel_mask,
                                                                           text_data[:, :pos_idx + 1],
                                                                           text_pos[:, :pos_idx + 1],
                                                                           text_mask[:, :pos_idx + 1, :pos_idx + 1])
                text_data[:, pos_idx] = text_out.argmax(-1)[:, pos_idx]
                if torch.sum(torch.sum(gate_out > .9, dim=1) > 0) == batch_size:
                    break

        return text_out, gate_out, att_heads_enc, att_heads_dec, att_heads

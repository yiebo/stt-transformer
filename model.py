import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import EncoderBlock, DecoderBlock


class PostNet(nn.Module):
    def __init__(self, mel_channels):
        super().__init__()
        conv_layers = []
        mapping = [mel_channels, 256, 256, 256, 256, mel_channels]
        for idx in range(len(mapping) - 1):
            block = nn.Sequential(nn.Conv1d(mapping[idx], mapping[idx + 1], kernel_size=5, padding=5 // 2),
                                  nn.BatchNorm1d(mapping[idx + 1]))
            conv_layers.append(block)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        x = x.transpose(1, 2)
        for conv in self.conv_layers[:-1]:
            x = conv(x)
            x = F.tanh(x)
            x = F.dropout(x, .5, self.training)
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
        nr_heads = 4
        head_channels = emb_channels // nr_heads
        self.pre_net = EncoderPre(in_channels, emb_channels)
        self.pos_alpha = nn.Parameter(torch.ones(1))

        self.enc_attention = nn.ModuleList([
            EncoderBlock(emb_channels, out_channels=emb_channels, emb_channels=head_channels, nr_heads=nr_heads),
            EncoderBlock(emb_channels, out_channels=emb_channels, emb_channels=head_channels, nr_heads=nr_heads),
            EncoderBlock(emb_channels, out_channels=emb_channels, emb_channels=head_channels, nr_heads=nr_heads)
        ])

    def forward(self, x, mask, pos_emb):
        # [B, T, C] -> [B, C, T]
        x = self.pre_net(x)
        x = x + self.pos_alpha * pos_emb
        x = F.dropout(x, .1, self.training)

        att_heads = []
        mask_ = mask.transpose(1, 2).unsqueeze(1)
        for att_block in self.enc_attention:
            x, att = att_block(x, mask)

            att = att.masked_fill(mask_, 0)
            att_heads.append(att)

        return x, tuple(att_heads)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, enc_channels, emb_channels):
        super().__init__()
        self.pre_net = nn.ModuleList([
            nn.Linear(in_channels, emb_channels),
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

        self.to_out = nn.Linear(emb_channels, out_channels)
        self.to_gate = nn.Linear(emb_channels, 1)

    def forward(self, x, enc_out, mask, enc_mask, pos_emb):
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
        dec_out_mask = mask[:, -1].unsqueeze(1).unsqueeze(-1)
        att_heads_dec, att_heads = [], []
        for att_block in self.dec_attention:
            x, att_dec, att_weight = att_block(x, enc_out, mask, enc_mask)
            x = F.leaky_relu(x)

            att_dec = att_dec.masked_fill(dec_out_mask, 0)
            att_weight = att_weight.masked_fill(dec_out_mask, 0)
            att_heads_dec.append(att_dec)
            att_heads.append(att_weight)

        text_out = self.to_out(x)
        gate_out = self.to_gate(x)

        # [B, T, 1]
        out_mask = mask[:, -1].unsqueeze(-1)
        text_out = text_out.masked_fill(out_mask, 0)

        gate_out = gate_out.masked_fill(out_mask, 1e3)
        gate_out = torch.sigmoid(gate_out)

        return text_out, gate_out, tuple(att_heads_dec), tuple(att_heads)

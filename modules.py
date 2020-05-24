import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import Attentionhead

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, nr_heads):
        super().__init__()
        self.attention = Attentionhead(in_channels, emb_channels, nr_heads)
        self.norm0 = nn.LayerNorm(in_channels)

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)

    def forward(self, x, mask):
        """
        args:
            x: [B, T, C], mask: [B, 1, T]
        ----
        return:
            x: [B, T, C], att_weight: [B, N, T, T]
        """
        # [B, T, C]
        att_out, att_weight = self.attention(query=x, key=x, value=x, mask=mask)
        x = self.norm0(x + att_out)

        x_ = self.linear(x)
        x = self.norm1(x + x_)
        return x, att_weight

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, enc_channels, out_channels, emb_channels, nr_heads):
        super().__init__()
        self.self_attention = Attentionhead(in_channels, emb_channels, nr_heads)
        self.norm0 = nn.LayerNorm(in_channels)

        self.attention = Attentionhead([in_channels, enc_channels, enc_channels], emb_channels, nr_heads)
        self.norm1 = nn.LayerNorm(in_channels)

        self.linear = nn.Sequential(
            nn.Linear(in_channels, 4 * out_channels),
            nn.ReLU(),
            nn.Linear(4 * out_channels, out_channels)
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x, enc_out, mask, enc_mask):
        """
        args:
            x: [B, T, C], enc_out: [B, T, C]
            mask: [B, T, T_enc], enc_mask: [B, 1, T_enc]
        ----
        return:
            x: [B, T, C]
            att_weight_self: [B, N, T, T], att_weight: [B, N, T, T_enc]
        """
        att_out, att_weight_self = self.self_attention(query=x, key=x, value=x, mask=mask)
        x = self.norm0(x + att_out)

        att_out, att_weight = self.attention(query=x, key=enc_out, value=enc_out, mask=enc_mask)
        x = self.norm1(x + att_out)

        x_ = self.linear(x)
        x = self.norm2(x + x_)

        return x, att_weight_self, att_weight

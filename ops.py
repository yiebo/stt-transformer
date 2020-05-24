import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleMod(nn.Module):
    def __init__(self, channels, latent):
        super().__init__()
        self.style_scale = LinearEqualized(latent, channels, gain=1.0)
        self.style_bias = LinearEqualized(latent, channels, gain=1.0)
        nn.init.ones_(self.style_scale.bias)

    def forward(self, x, latent):
        style_scale = self.style_scale(latent).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(latent).unsqueeze(2).unsqueeze(3)
        x = x * style_scale + style_bias
        return x

class Conv2d_AdaIn(nn.Conv2d):
    def __init__(self, latent_channels, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2d_AdaIn, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)
        self.style_scale = nn.Linear(latent_channels, out_channels)

    def forward(self, x, latent):
        batch_size = x.shape[0]
        # modulate
        # weight = self.weight  #[1,O,I,K,K]

        # weight *= self.style_scale(latent).unsqueeze(2).unsqueeze(3).unsqueeze(4) #[B,O,I,K,K]
        style = self.style_scale(latent).unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B,O,I,K,K]

        weight = (self.weight + 1.) * style

        # demod
        style_std = torch.rsqrt(torch.sum(weight.pow(2), dim=[2, 3, 4], keepdim=True) + 1e-8)  # [B,O,1,1,1]
        weight = weight * style_std  # [B,O,I,K,K]

        weight = weight.view(-1, weight.shape[2], weight.shape[3], weight.shape[4])  # [BO,I,K,K]
        x = x.view(1, -1, x.shape[2], x.shape[3])  # [1,BC,H,W]

        x = F.conv2d(x, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=batch_size)
        # print(x.shape)
        x = x.view(-1, self.weight.shape[0], x.shape[2], x.shape[3])

        return x + self.bias.unsqueeze(1).unsqueeze(2)


class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        # self, in_channels, out_channels, scale, kernel_size=3, stride=1, padding=1):
        super().__init__()
        out_channels = out_channels * scale ** 2
        self.sub_pixel_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias, padding_mode),
                                            nn.PixelShuffle(scale))

    def forward(self, x):
        x = self.sub_pixel_conv(x)
        return x


class Conv2dEqualized(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, gain=2 ** .5):
        super().__init__()
        self.w_scale = gain / np.sqrt(in_channels * kernel_size ** 2)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        weight = self.w_scale * self.weight
        bias = self.bias
        if bias is not None:
            bias = self.w_scale * bias
        return F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding)

class LinearEqualized(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, gain=2 ** .5):
        super().__init__()

        self.w_scale = gain / np.sqrt(in_channels)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        weight = self.w_scale * self.weight
        bias = self.bias
        if bias is not None:
            bias = self.w_scale * bias
        return F.linear(x, weight, bias)

class Noise(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


def attention(query, key, value, mask, scale=1):
    """"
    args:
        query, key value: [B, N, T, C]
    ----
    return:
        att_out: [B, N, T_query, C]
        att_weight: [B, N, T_query, T_kv]
    """
    # [B, N, T_q, C] * [B, N, C, T_kv] -> [B, N, T_q, T_kv]
    att_weight = torch.matmul(query, key.transpose(2, 3))

    att_weight = scale * att_weight
    att_weight = torch.masked_fill(att_weight, mask, -float('Inf'))

    att_weight = F.softmax(att_weight, -1)
    # [B, N, T_q, T_kv] * [B, N, T_kv, C] -> [B, N, T_q, C]
    att_out = torch.matmul(att_weight, value)
    return att_out, att_weight


class Attentionhead(nn.Module):
    def __init__(self, in_channels, emb_channels, nr_heads):
        super().__init__()
        if isinstance(in_channels, list):
            q_channels, k_channels, v_channels = in_channels
        else:
            q_channels = k_channels = v_channels = in_channels
        self.nr_heads = nr_heads
        self.emb_channels = emb_channels
        self.scale = emb_channels ** -.5
        self.to_query = nn.Linear(q_channels, nr_heads * emb_channels)
        self.to_key = nn.Linear(k_channels, nr_heads * emb_channels)
        self.to_value = nn.Linear(v_channels, nr_heads * emb_channels)

        self.to_out = nn.Linear(nr_heads * emb_channels, q_channels)

    def forward(self, query, key, value, mask):
        """
        args:
            query, key, value: [B, T, C]
            mask: [B, T_query, T_value]
        -----
        return:
            att_out: [B, T_query, C]
            att_weight: [B, N, T_query, T_kv]
        """
        # [B, T, C] -> [B, T, N*C] -> [B, T, N, C] -> [B, N, T, C]
        query = self.to_query(query).view(-1, query.size(1), self.nr_heads, self.emb_channels)
        key = self.to_key(key).view(-1, key.size(1), self.nr_heads, self.emb_channels)
        value = self.to_value(value).view(-1, value.size(1), self.nr_heads, self.emb_channels)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        mask = mask.view(-1, 1, mask.size(1), mask.size(2))

        att_out, att_weight = attention(query, key, value, mask, self.scale)

        # [B, N, T_query, C] -> [B, T_query, N, C] -> [B, T_query, N*C]
        att_out = att_out.transpose(1, 2).contiguous()
        att_out = att_out.view(-1, att_out.size(1), self.nr_heads * self.emb_channels)
        att_out = self.to_out(att_out)

        return att_out, att_weight

def positional_encoding(max_len, emb_channels):
    max_len = max_len - 1
    pe = torch.zeros(max_len, emb_channels)
    # [max_len, 1]
    position = torch.arange(0., max_len).unsqueeze(1)
    # [emb_channels // 2]
    div_term = max_len ** (torch.arange(0, emb_channels, 2.) / emb_channels)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)

    padding_encoding = torch.zeros(1, emb_channels)
    pe = torch.cat([padding_encoding, pe], 0)

    return pe

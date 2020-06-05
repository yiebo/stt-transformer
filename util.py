import torch
import torch.nn.functional as F
from torchvision import utils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# plt.switch_backend('agg')
from dataset import _id_to_symbol

def plot_heatmap(input, x_label=None, y_label=None):
    # Set up figure with colorbar
    input = input.detach().to('cpu').numpy()
    batches = input.shape[0]
    fig, axs = plt.subplots(batches, sharex=True, sharey=True)
    for input_, ax in zip(input, axs):
        cax = ax.imshow(input_, interpolation='nearest')
        fig.colorbar(cax, ax=ax)
        # Set up axes
        if x_label is not None:
            ax.set_xticklabels(x_label, rotation=90)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        if y_label is not None:
            y_label = [_id_to_symbol[id_] for id_ in x_label]
            ax.set_yticklabels(y_label)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout(pad=1)
    return fig


def confusion_matrix(input, target, labels, ignore=-1):
    input, target = input.detach().to('cpu'), target.detach().to('cpu')
    matrix = torch.zeros(len(labels), len(labels))
    count = torch.zeros(len(labels), 1)
    input = F.softmax(input, -1)
    for batch_idx in range(input.size(0)):
        for input_, target_ in zip(input[batch_idx], target[batch_idx]):
            if target_ == ignore:
                break
            matrix[target_] = matrix[target_] + input_
            count[target_] = count[target_] + 1
    count = count.clamp(1)
    matrix = matrix / count

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy(), interpolation='nearest')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + labels, rotation=90)
    ax.set_yticklabels([''] + labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    return fig

def plot_att_heads(att_heads, batch_idx=0, combine=True):
    """
    att_heads: L * [B, N, T, T]
    """
    heads = []
    max_heads = max([head.size(1) for head in att_heads])
    for layer_att_heads in att_heads:
        head_size = layer_att_heads.size(1)
        if head_size < max_heads:
            layer_att_heads = torch.nn.functional.pad(layer_att_heads,
                                                      [0, 0, 0, 0, 0, max_heads - head_size])
        layer_att_heads = layer_att_heads[batch_idx].transpose(1, 2).unsqueeze(1)
        if combine:
            layer_att_heads = torch.cat([layer_att_heads.sum(0, keepdim=True), layer_att_heads], 0)
        layer_att_heads_out = utils.make_grid(layer_att_heads, nrow=1, padding=2, pad_value=.5,
                                              normalize=True, scale_each=True)
        heads.append(layer_att_heads_out)
    return torch.cat(heads, 2)

def text_id_to_string(text, text_len=None, ignore=-1):
    text_out = ['']
    if text_len is None:
        text_len = torch.tensor(text.size(0) * [text.size(1)])
    for text_, text_len_ in zip(text, text_len):
        text_list = []
        for idx in range(text_len_):
            char_id = text_[idx].item()
            if char_id == ignore:
                break
            text_list.append(_id_to_symbol[char_id])
        text_list = [f'{len(text_list)} | '] + text_list
        text_string = ''.join(text_list)
        text_out.append(text_string)
    text_out = '    \n'.join(text_out)
    return text_out

def to_device(tensors: list, device='cuda:0'):
    return [tensor.to(device) for tensor in tensors]

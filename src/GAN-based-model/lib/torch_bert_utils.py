import math

import torch
from torch import nn
import numpy as np

from .torch_utils import get_tensor_from_array


def get_attention_mask(lens: np.array, max_len: int):
    """
    :param lens: shape (N,)
    convert sequence lengths to sequence masks
    mask: shape:(N, T)
    """
    lens = torch.Tensor(lens).long()
    mask = (torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)).float()
    mask = get_tensor_from_array(mask)
    return mask


def get_seq_mask(inp: torch.Tensor, mask_prob: float):
    """
    create mask for mask-lm
    :return: shape: (N, T) or (N, T, 1) according to rank of inp
    0: masked,
    doesn't take care of the padding at the end
    """
    if inp.ndimension() == 3:
        rank2 = inp[:, :, 0]
    else:
        rank2 = inp
    mask = (torch.empty_like(rank2, dtype=torch.float).uniform_() > mask_prob).float()
    if inp.ndimension() == 3:
        mask = mask.unsqueeze(2)
    return mask


def get_mlm_masks(inp: torch.Tensor, mask_prob, mask_but_no_prob):
    """
    :param inp: the tensor to be masked
    :param mask_prob: mask probability at prediction
    :param mask_but_no_prob: the ratio of reconstruction
    :return:
    input_mask: mask to be applied to the input. masking positions are 0s
    predict_mask: positions to predict, indicated with zeros
    See tests for more info.
    """
    predict_mask = get_seq_mask(inp, mask_prob)  # mask_prob of 0s
    temp_mask = get_seq_mask(inp, mask_but_no_prob)
    input_mask = 1 - (1 - predict_mask) * temp_mask  # fewer 0s
    return input_mask, predict_mask


def get_sep_mask(inp: torch.Tensor, lens: np.array):
    mask = np.zeros([inp.shape[0], inp.shape[1]])
    positions = np.clip(lens, a_max=inp.shape[1] - 1, a_min=None)  # shape: (N,)
    mask[np.arange(mask.shape[0], dtype=int), positions] = 1
    mask = get_tensor_from_array(mask)
    return mask


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, emb_size, dropout=0., max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size, dtype=torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.float() + self.pe[:, :x.size(1)]
        return self.dropout(x)

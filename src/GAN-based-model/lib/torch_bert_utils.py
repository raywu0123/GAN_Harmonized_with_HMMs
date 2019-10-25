import torch
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


def get_mlm_masks(cls, inp: torch.Tensor, mask_prob, mask_but_no_prob):
    predict_mask = cls.get_seq_mask(inp, mask_prob)  # mask_prob of 0s
    temp_mask = cls.get_seq_mask(inp, mask_but_no_prob)
    input_mask = 1 - (1 - predict_mask) * temp_mask  # fewer 0s
    return input_mask, predict_mask

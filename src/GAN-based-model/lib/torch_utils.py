import torch
from torch import nn
import numpy as np

epsilon = 1e-12


def get_tensor_from_array(arr: np.array) -> torch.Tensor:
    arr = torch.Tensor(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr


def masked_reduce_mean(arr: torch.Tensor, mask: torch.Tensor):
    assert(arr.shape[:2] == mask.shape[:2])
    if arr.ndimension() == 3:
        mask = mask.unsqueeze(2)
    arr = arr * mask
    return arr.sum(dim=1) / (mask.sum(dim=1) + epsilon)


def masked_reduce_sum(arr: torch.Tensor, mask: torch.Tensor):
    assert(arr.shape[:2] == mask.shape[:2])
    if arr.ndimension() == 3:
        mask = mask.unsqueeze(2)
    arr = arr * mask
    return arr.sum(dim=1)


def l2_loss(pred, tar, mask):
    assert(pred.shape == tar.shape)
    mask_l2_loss = torch.sum((pred - tar) ** 2, dim=2)  # shape: (N, T)
    loss = torch.mean(masked_reduce_mean(mask_l2_loss, mask))
    return loss


def cpc_loss(pred, tar, pred_mask, attention_mask):
    """
    :param pred: shape: (N, T, V)
    :param tar:  shape: (N, L, V)
    :param pred_mask: shape: (N, T), ones at where to predict
    :param attention_mask: shape: (N, T), ones at non-padding timesteps
    assert T = L

    cpc_loss = -log(exp_pos_inner_product / sum(exp_neg_inner_product) + exp_pos_inner_product)
    = log((exp_pos_inner_product + sum(exp_neg_inner_product)) / exp_pos_inner_product)
    """
    assert(pred.shape == tar.shape)
    pred = pred * attention_mask.unsqueeze(2)
    tar = tar * attention_mask.unsqueeze(2)
    inner_products = torch.einsum('nte,nle->nlt', [pred, tar])  # shape: (N, L, T)
    other_sample_neg_inner_products = torch.einsum('nte,nle->nlt', [pred, torch.flip(tar, dims=[0])])
    max_logit = max(torch.max(inner_products), torch.max(other_sample_neg_inner_products))
    inner_products = inner_products - max_logit + 80
    other_sample_neg_inner_products = other_sample_neg_inner_products - max_logit + 80

    exp_inner_products = torch.exp(inner_products)
    exp_inner_products = exp_inner_products * attention_mask.unsqueeze(2)
    sum_exp_inner_product = masked_reduce_sum(exp_inner_products, attention_mask)  # shape: (N, T)
    exp_pos_inner_product = torch.diagonal(exp_inner_products, dim1=1, dim2=2) * attention_mask
    # shape: (N, T)
    mean_neg_exp_inner_product = \
        (sum_exp_inner_product - exp_pos_inner_product) / (torch.sum(attention_mask, dim=-1, keepdim=True) - 1 + epsilon)
    other_mean_neg_exp_inner_product = masked_reduce_mean(
        torch.exp(other_sample_neg_inner_products),
        torch.flip(attention_mask, dims=[0]),
    )  # shape: (N, T)
    mean_neg_exp_inner_product = (mean_neg_exp_inner_product + other_mean_neg_exp_inner_product) / 2.
    pos_inner_products = torch.diagonal(inner_products, dim1=1, dim2=2) * attention_mask
    sample_cpc = masked_reduce_mean(
        torch.log(exp_pos_inner_product + mean_neg_exp_inner_product + epsilon) - pos_inner_products,
        pred_mask,
    )
    cpc = sample_cpc.mean()
    return cpc


def intra_segment_loss(logits, repeats, mask, sep_size):
    probs = torch.softmax(logits, dim=-1)
    start_prob = probs[:sep_size]
    end_prob = probs[sep_size:]
    partial_mask = mask[:sep_size]
    error = torch.sum((start_prob - end_prob) ** 2, dim=-1)
    return torch.sum(masked_reduce_sum(error, partial_mask)) / (torch.sum(repeats) + epsilon)


def inter_segment_loss(logits, mask):
    """
    Implement Jensen-Shannon Divergence for multiple distributions
    JSD = H(sum(Pi)/m) - H(Pi)/m
    """
    probs = torch.softmax(logits, dim=-1)  # (N, T, V)
    mean_probs = masked_reduce_mean(probs, mask)  # (N, V)
    H_of_mean_probs = torch.sum(-mean_probs * torch.log(mean_probs + epsilon), dim=-1)  # (N,)

    Hs = torch.sum(-probs * torch.log(probs + epsilon), dim=-1)  # (N, T)
    mean_of_Hs = masked_reduce_mean(Hs, mask)  # (N,)
    JSDs = H_of_mean_probs - mean_of_Hs
    JSD = torch.mean(JSDs)
    return -JSD


def gumbel_sample(logits):
    U = torch.empty_like(logits).uniform_(epsilon, 1 - epsilon)
    gumbel_noise = -torch.log(-torch.log(U))
    logits = logits + gumbel_noise
    return torch.argmax(logits, dim=-1)


class EMA(nn.Module):

    def __init__(self, decay_rate):
        super(EMA, self).__init__()
        self.decay_rate = decay_rate
        self.decay_rate_power = 1
        self.average = nn.Parameter(torch.Tensor([0]), requires_grad=False)

    def forward(self, x):
        self.decay_rate_power *= self.decay_rate
        self.average.data = self.decay_rate * self.average + (1 - self.decay_rate) * x
        return self.average / (1 - self.decay_rate_power)


def create_attention_mask(lens: np.array, max_len: int):
    """
    :param lens: shape (N,)
    convert sequence lengths to sequence masks
    mask: shape:(N, T)
    """
    lens = torch.Tensor(lens).long()
    mask = (torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)).float()
    mask = get_tensor_from_array(mask).detach()
    return mask


def first_order_expand(grad, word_vecs, embeddings):
    first_order_y = torch.einsum('nte,ve->ntv', [grad, embeddings])
    first_order_y -= torch.sum(grad * word_vecs, dim=-1, keepdim=True)
    return first_order_y

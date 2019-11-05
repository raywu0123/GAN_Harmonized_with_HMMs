from .uns import UnsModel
from .sup import SupModel
from .uns_bert import UnsBertModel
from .swap_gan import SwapGANModel
from .data_efficient_bert import DataEfficientBert
from .segment_gan import SegmentGANModel


MODEL_HUB = {
    'uns': UnsModel,
    'sup': SupModel,
    'uns_bert': UnsBertModel,
    'swap_gan': SwapGANModel,
    'data_efficient_bert': DataEfficientBert,
    'segment_gan': SegmentGANModel,
}

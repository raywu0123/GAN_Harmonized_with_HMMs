from .uns import UnsModel
from .sup import SupModel
from .uns_bert import UnsBertModel
from .swap_gan import SwapGANModel
from .data_efficient_bert import DataEfficientBert
from .segment_gan import SegmentGANModel
from .sup_segment import SupervisedSegmentModel


MODEL_HUB = {
    'uns': UnsModel,
    'sup': SupModel,
    'uns_bert': UnsBertModel,
    'swap_gan': SwapGANModel,
    'data_efficient_bert': DataEfficientBert,
    'sup_segment': SupervisedSegmentModel,
    'segment_gan': SegmentGANModel,
}

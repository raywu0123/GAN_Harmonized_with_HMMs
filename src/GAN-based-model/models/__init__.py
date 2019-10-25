from .uns import UnsModel
from .sup import SupModel
from .uns_bert import UnsBertModel
from .swap_gan import SwapGANModel


MODEL_HUB = {
    'uns': UnsModel,
    'sup': SupModel,
    'uns_bert': UnsBertModel,
    'swap_gan': SwapGANModel,
}

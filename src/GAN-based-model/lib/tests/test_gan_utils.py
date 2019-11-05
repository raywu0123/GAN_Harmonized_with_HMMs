from unittest import TestCase

import torch

from lib.torch_gan_utils import (
    to_onehot,
)


class TestGANUtils(TestCase):

    def test_to_onehot(self):
        tensor = torch.Tensor([
            [0, 1, 2],
            [0, 2, 1],
            [2, 1, 0],
        ])
        tensor = tensor.long()
        onehot_ans = torch.Tensor([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ])
        self.assertTrue(
            torch.all(to_onehot(tensor, 3) == onehot_ans).item() == 1
        )





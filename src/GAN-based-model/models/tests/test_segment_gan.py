import torch
import numpy as np

from ..segment_gan import SegmentGANModel, SegmentMapGenerator
from ..sup_segment import SupervisedSegmentModel


def test_get_selected_feats():
    batch_feats = torch.Tensor([
        [[1], [2], [3], [4], [5], [6], [0], [0]],
        [[9], [8], [7], [6], [0], [0], [0], [0]],
    ]).float()
    feat_lens = torch.Tensor([
        6, 4
    ])
    feat_indices = torch.Tensor([
        [0, 0.3, 2.3, 7.3],
        [0, 1.2, 5., 100.],
    ])

    out, _ = SegmentGANModel.get_selected_feats(batch_feats, feat_lens, feat_indices)
    expected = torch.Tensor([
        [[1], [1.3], [3.3], [0.]],
        [[9], [7.8], [0.], [0.]]
    ])
    assert (out == expected).all().item()


def test_process_orc_bnd():
    feat_max_length = 100
    phn_max_length = 20
    bnds = [
        [0, 20, 25, 35, 50]
    ]
    out = SupervisedSegmentModel.process_orc_bnd(bnds, phn_max_length, feat_max_length)


def test_non_maximum_suppression():
    x = torch.Tensor([
        [
            [1.5, -0.3, 1.2, 2.0, 1.0, 0., 1.3],
            [4, 4, 4, 4, 3, 3, 3]
        ],
    ])
    phn_max_length = 3
    centers = SegmentMapGenerator.non_maximum_suppression(x, phn_max_length)
    centers = centers.data.numpy()

    expected = np.array([[5, 0.7, 7]])
    np.testing.assert_array_almost_equal(centers, expected)

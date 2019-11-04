from unittest import TestCase

import numpy as np

from evalution import phn_eval


class TestPhnEval(TestCase):

    def setUp(self) -> None:
        self.vocab_size = 26
        self.N, self.T = 100, 128
        self.phn_mapping = {
            i: chr(ord('a') + i)
            for i in range(self.vocab_size)
        }

    def test_same(self):
        pred = np.random.randint(0, self.vocab_size, size=[self.N, self.T])
        label = pred
        lens = np.random.randint(0, self.T, size=[self.N])
        error_count, label_length, sample_labels, sample_preds = phn_eval(
            pred, lens, label, self.phn_mapping,
        )
        self.assertEqual(error_count, 0)
        self.assertLessEqual(label_length, np.sum(lens))

    # def test_random(self):
    #     pred = np.random.randint(0, self.vocab_size, size=[self.N, self.T])
    #     label = np.random.randint(0, self.vocab_size, size=[self.N, self.T])
    #     lens = np.random.randint(0, self.T, size=[self.N])
    #     error_count, label_length, sample_labels, sample_preds = phn_eval(
    #         pred, lens, label, self.phn_mapping,
    #     )
    #     self.assertAlmostEqual(error_count / label_length, 1., places=2)

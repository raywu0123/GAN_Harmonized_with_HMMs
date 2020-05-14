import sys

import numpy as np
import torch

from .base import ModelBase
from .segment_gan import SegmentMapGenerator
from lib.torch_utils import get_tensor_from_array, masked_reduce_mean
from callbacks import Logger


class SupervisedSegmentModel(ModelBase):

    description = "SUPERVISED-SEGMENT MODEL"

    def __init__(self, config):
        self.config = config

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.segment_map_generator = SegmentMapGenerator(config, self.config.phn_max_length)

        self.opt = torch.optim.Adam(
            params=self.segment_map_generator.parameters(),
            lr=config.gen_lr,
            betas=[0.5, 0.9],
        )
        if torch.cuda.is_available():
            self.segment_map_generator.cuda()

        sys.stdout.write('\b' * len(cout_word))
        cout_word = f'{self.description}: finish     '
        sys.stdout.write(cout_word + '\n')
        sys.stdout.flush()

    def train(
            self,
            config,
            data_loader,
            dev_data_loader=None,
            aug=False,
    ):
        print('TRAINING(unsupervised)...')
        batch_size = config.batch_size * config.repeat
        logger = Logger(print_step=config.print_step)
        step = 1
        for epoch in range(1, config.epoch + 1):
            for batch in data_loader.get_batch(batch_size):
                self.train_segment_mapper(batch, logger)

                if step % config.eval_step == 0:
                    eval_loss, eval_centers_mse = self.eval(dev_data_loader, batch_size)
                    logger.update({
                        "eval_mse": eval_loss,
                        "eval_centers_mse": eval_centers_mse,
                    }, group_name="sup-segment-signals")
                logger.step()
                step += 1
        print('=' * 80)

    def train_segment_mapper(self, batch, logger):
        self.opt.zero_grad()
        feats, orc_bnd, source_lens = batch['source'], batch['source_bnd'], batch['source_length']
        batch_feats = get_tensor_from_array(feats)

        processed = self.process_orc_bnd(orc_bnd, self.config.phn_max_length, self.config.feat_max_length)
        dist_to_centers, widths, feat_mask = processed['dis_to_centers'], processed['widths'], processed['feat_mask']
        centers_label, phn_mask = processed['centers'], processed['phn_mask']

        label = torch.cat([dist_to_centers.unsqueeze(1), widths.unsqueeze(1)], dim=1)  # (N, 2, feat_mask_length)
        scaled_label = label / self.config.feat_max_length
        pred = self.segment_map_generator._forward(batch_feats, feat_mask)
        scaled_pred = pred / self.config.feat_max_length

        loss = masked_reduce_mean(((scaled_label - scaled_pred) ** 2).transpose(1, 2), feat_mask).mean()
        loss.backward()
        self.opt.step()

        centers_pred = self.segment_map_generator.non_maximum_suppression(pred, self.config.phn_max_length, feat_mask)
        centers_mse = masked_reduce_mean((centers_pred - centers_label) ** 2, phn_mask).mean()

        logger.update({
            "train_mse": loss.item() * self.config.feat_max_length ** 2,
            'train_centers_mse': centers_mse.item(),
        }, group_name="sup-segment-signals")

    @staticmethod
    def process_orc_bnd(orc_bnd: np.array, n_param: int, feat_max_length: int):
        out = np.ones([len(orc_bnd), n_param])
        phn_mask = np.zeros_like(out)

        dis_to_centers = np.zeros([len(orc_bnd), feat_max_length])
        all_widths = np.zeros_like(dis_to_centers)
        feat_mask = np.zeros_like(dis_to_centers)

        for idx, bnd in enumerate(orc_bnd):
            bnd = np.array(bnd)
            widths = bnd[1:] - bnd[:-1]
            centers = (bnd[1:] + bnd[:-1]) / 2
            out[idx, :len(centers)] = centers
            for w, c in zip(widths, centers):
                for i in range(w):
                    frame_index = int(c + (i - w / 2))
                    displacement = i - w / 2
                    dis_to_centers[idx, frame_index] = displacement
                    all_widths[idx, frame_index] = w

            phn_mask[idx, :len(centers)] = 1
            feat_mask[idx, :bnd[-1]] = 1

        return {
            "dis_to_centers": get_tensor_from_array(dis_to_centers),
            'widths': get_tensor_from_array(all_widths),
            "feat_mask": get_tensor_from_array(feat_mask),
            "centers": get_tensor_from_array(out),
            "phn_mask": get_tensor_from_array(phn_mask),
        }

    def eval(self, data_loader, batch_size):
        losses = []
        centers_mses = []
        with torch.no_grad():
            for batch in data_loader.get_batch(batch_size):
                feats, orc_bnd, source_lens = batch['source'], batch['source_bnd'], batch['source_length']
                batch_feats = get_tensor_from_array(feats)

                processed = self.process_orc_bnd(orc_bnd, self.config.phn_max_length, self.config.feat_max_length)
                dist_to_centers, widths, feat_mask = processed['dis_to_centers'], processed['widths'], processed[
                    'feat_mask']
                centers_label, phn_mask = processed['centers'], processed['phn_mask']

                label = torch.cat([dist_to_centers.unsqueeze(1), widths.unsqueeze(1)],
                                  dim=1)  # (N, 2, feat_mask_length)
                scaled_label = label / self.config.feat_max_length
                pred = self.segment_map_generator._forward(batch_feats, feat_mask)
                scaled_pred = pred / self.config.feat_max_length

                loss = masked_reduce_mean(((scaled_label - scaled_pred) ** 2).transpose(1, 2), feat_mask).mean()
                losses.append(loss.item() * self.config.feat_max_length ** 2)

                centers_pred = self.segment_map_generator.non_maximum_suppression(
                    pred,
                    self.config.phn_max_length,
                    feat_mask,
                )
                centers_mse = masked_reduce_mean((centers_pred - centers_label) ** 2, phn_mask).mean()
                centers_mses.append(centers_mse.item())

        return np.mean(losses), np.mean(centers_mses)

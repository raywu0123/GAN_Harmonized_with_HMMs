import sys
import os

import numpy as np
from scipy.stats.stats import pearsonr
import torch
from torch import nn
from torch.nn import functional as F

from .base import ModelBase
from lib.torch_utils import get_tensor_from_array
from lib.torch_gan_utils import (
    to_onehot,
    compute_gradient_penalty,
)
from lib.utils import array_to_string
from lib.torch_bert_utils import get_attention_mask
from lib.torch_utils import masked_reduce_mean
from evalution import frame_eval
from callbacks import Logger


class SegmentGANModel(ModelBase):

    description = "SEGMENT-GAN MODEL"

    def __init__(self, config):
        self.config = config

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.generator = Generator(config)
        self.segment_mapper = SegmentMapper(config)
        self.segment_map_generator = SegmentMapGenerator(config, self.segment_mapper.n_param)
        self.critic = Critic(config)

        self.g_opt = torch.optim.Adam(
            params=list(self.generator.parameters()) + list(self.segment_map_generator.parameters()),
            lr=config.gen_lr,
            betas=[0.5, 0.9],
        )
        self.c_opt = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=config.dis_lr,
            betas=[0.5, 0.9],
        )

        if torch.cuda.is_available():
            self.generator.cuda()
            self.segment_map_generator.cuda()
            self.segment_mapper.cuda()
            self.critic.cuda()

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
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

        batch_size = config.batch_size * config.repeat
        logger = Logger(print_step=config.print_step)
        max_err = 100.0
        frame_temp = 0.9
        step = 1
        for epoch in range(1, config.epoch + 1):
            for batch in data_loader.get_batch(batch_size):
                if step == 8000:
                    frame_temp = 0.8
                if step == 12000:
                    frame_temp = 0.7

                if step % (config.dis_iter + config.gen_iter) < config.dis_iter:
                    self.generator.eval()
                    self.critic.train()
                    self.train_critic(batch, get_target_batch, logger, config, frame_temp)
                else:
                    self.critic.eval()
                    self.generator.train()
                    self.train_generator(batch, logger, frame_temp)

                if step % config.eval_step == 0:
                    step_err = frame_eval(self.predict_batch, dev_data_loader)
                    print(f'EVAL max: {max_err:.2f} step: {step_err:.2f}')
                    logger.update({'fer': step_err}, ema=False, group_name='val')
                    if step_err < max_err:
                        max_err = step_err
                        self.save(config.save_path)
                logger.step()
                step += 1
        print('=' * 80)

    def generate(self, batch):
        feats, feat_lens = batch['source'], batch['source_length']
        feat_mask = get_attention_mask(feat_lens, self.config.feat_max_length)
        batch_feats, feat_lens = get_tensor_from_array(feats), get_tensor_from_array(feat_lens)

        segment_map_params = self.segment_map_generator(batch_feats, feat_mask)
        feat_indices = self.segment_mapper.get_feat_indices(segment_map_params)  # (batch_size, config.phn_max_length)
        selected_feats, mask = self.get_selected_feats(batch_feats, feat_lens, feat_indices)  # (batch_size, config.phn_max_length)
        fake_target_logits = self.generator(selected_feats)

        noisy_feat_indices = feat_indices + torch.empty_like(feat_indices).to(feat_indices.device).uniform_(-1, 1)
        noisy_selected_feats, _ = self.get_selected_feats(batch_feats, feat_lens, noisy_feat_indices)
        noisy_fake_target_logits = self.generator(noisy_selected_feats)
        return {
            "fake_target_logits": fake_target_logits,
            "noisy_fake_target_logits": noisy_fake_target_logits,
            "mask": mask,
        }

    @staticmethod
    def get_selected_feats(batch_feats: torch.Tensor, feat_lens, feat_indices: torch.Tensor):
        """interpolation"""
        batch_size = len(batch_feats)
        feat_max_length = batch_feats.shape[1]
        phn_max_length = feat_indices.shape[1]
        all_indices = torch.arange(feat_max_length).to(batch_feats.device).float().repeat((batch_size, phn_max_length, 1))
        # (batch_size, phn_max_length, feat_max_length)
        selected_indices = feat_indices.repeat(feat_max_length, 1, 1).permute([1, 2, 0])  # (batch_size, phn_max_length, feat_max_length)
        interpolation_weights = (1 - (selected_indices - all_indices).abs()).clamp(max=1., min=0.)
        interpolation = torch.matmul(interpolation_weights, batch_feats)  # (batch_size, phn_max_length, feat_dim)

        mask = (feat_indices <= feat_lens.unsqueeze(1)).float().unsqueeze(2)  # (batch_size, phn_max_length, 1)
        mask[:, 0] = 1
        interpolation = interpolation * mask
        return interpolation, mask

    def train_critic(self, batch, get_target_batch, logger, config, frame_temp):
        self.c_opt.zero_grad()

        gen = self.generate(batch)
        fake_target_logits, mask = gen['fake_target_logits'], gen['mask']
        fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
        fake_score = self.critic(fake_target_probs, mask).mean()

        batch_size = len(batch['source'])
        real_target_idx, batch_target_len = get_target_batch(batch_size)
        real_target_idx = get_tensor_from_array(real_target_idx).long()
        real_target_probs = to_onehot(real_target_idx, class_num=config.phn_size)
        mask = get_attention_mask(batch_target_len, real_target_probs.shape[1]).unsqueeze(2)
        real_score = self.critic(real_target_probs, mask).mean()

        inter_alphas = torch.empty(
            [fake_target_probs.shape[0], 1, 1],
            device=fake_target_probs.device,
        ).uniform_()
        inter_samples = real_target_probs + inter_alphas * (fake_target_probs.detach() - real_target_probs)
        inter_samples = torch.tensor(inter_samples, requires_grad=True)
        inter_score = self.critic(inter_samples)  # shape: (N, 1)
        gradient_penalty = compute_gradient_penalty(inter_score, inter_samples)

        c_loss = -real_score + fake_score + config.penalty_ratio * gradient_penalty
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.)
        self.c_opt.step()
        logger.update({
            'c_loss': c_loss.item(),
            'gradient_penalty': gradient_penalty.item(),
        }, group_name='GAN_losses')
        logger.update({
            'true_sample': array_to_string(real_target_idx[0].cpu().data.numpy()),
            'mean_true_length': batch_target_len.mean(),
            'std_true_length': batch_target_len.std(),
        }, group_name="signals")

    def train_generator(self, batch, logger, frame_temp):
        self.g_opt.zero_grad()

        gen = self.generate(batch)
        fake_target_logits, noisy_fake_target_logits, mask = gen['fake_target_logits'], gen['noisy_fake_target_logits'], gen['mask']

        fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
        fake_score = self.critic(fake_target_probs, mask).mean()  # shape: (N, 1)

        noisy_fake_target_probs = F.softmax(noisy_fake_target_logits / frame_temp, dim=-1)
        segment_loss = ((fake_target_probs - noisy_fake_target_probs) ** 2).sum(-1).mean()

        g_loss = -fake_score
        total_loss = g_loss + self.config.seg_loss_ratio * segment_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.generator.parameters()) + list(self.segment_map_generator.parameters()),
            5.
        )
        self.g_opt.step()

        mask = mask.squeeze()  # (N, T)
        lengths = mask.sum(dim=-1).cpu().data.numpy()  # (N,)
        true_lengths = np.array([len(bnd) for bnd in batch['source_bnd']], dtype=float)
        length_correlation = pearsonr(lengths, true_lengths)[0]

        fake_sample = torch.argmax(fake_target_probs[0], dim=-1).long() * mask[0].long()
        logger.update({
            'g_loss': g_loss.item(),
        }, group_name='GAN_losses')
        logger.update({
            'seg_loss': segment_loss.item(),
        }, group_name='segment_losses')
        logger.update({
            'fake_sample': array_to_string(fake_sample.cpu().data.numpy()),
            'mean_fake_length': np.mean(lengths),
            'std_fake_length': np.std(lengths),
            'length_correlation': length_correlation
        }, group_name='signals')

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # torch.save(
        #     {
        #     },
        #     os.path.join(save_path, 'checkpoint.pth.tar')
        # )

    def restore(self, save_dir):
        # TODO
        pass

    def predict_batch(self, batch_feat, batch_len):
        self.generator.eval()
        with torch.no_grad():
            batch_feat = get_tensor_from_array(batch_feat)
            batch_frame_logits = self.generator(batch_feat)
            batch_frame_prob = torch.softmax(batch_frame_logits, dim=-1)

        batch_frame_prob = batch_frame_prob.cpu().data.numpy()
        self.generator.train()
        return batch_frame_prob


class Generator(nn.Module):
    """
    Framewise mapping from feature to phoneme logits
    """
    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.feat_dim, config.gen_hidden_size)
        self.dense_out = nn.Linear(config.gen_hidden_size, config.phn_size)

    def forward(self, x, mask=None) -> torch.Tensor:
        x = self.dense_in(x)
        x = self.dense_out(x)
        return x


class Critic(nn.Module):

    def __init__(self, config, kernel_sizes=(3, 5, 7, 9)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Linear(config.phn_size, config.dis_emb_size, bias=False)
        self.first_channels = config.dis_emb_size
        self.first_convs = nn.ModuleList([
            nn.Conv1d(
                config.dis_emb_size,
                self.first_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        ])
        self.second_channels = config.dis_emb_size
        self.second_convs = nn.ModuleList([
            nn.Conv1d(
                self.first_channels * len(kernel_sizes),
                self.second_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        ])
        self.activation = torch.nn.LeakyReLU()
        self.dense = nn.Linear(config.phn_max_length * self.second_channels * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: (N, T, V)
        if mask is not None:
            x = x * mask

        e = self.embedding(x)
        N, T, E = e.shape

        e = e.view(N, E, T)
        outputs = torch.cat([
            self.activation(conv(e)) for conv in self.first_convs
        ], dim=1)
        outputs = torch.cat([
            self.activation(conv(outputs)) for conv in self.second_convs
        ], dim=1)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.dense(outputs)
        return outputs


class SegmentMapGenerator(nn.Module):

    def __init__(self, config, num_segment_map_param: int):
        super().__init__()
        self.embedding = nn.Linear(config.feat_dim, config.dis_emb_size, bias=False)
        self.num_segment_map_param = num_segment_map_param

        self.layers = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size, config.dis_emb_size // 2, kernel_size=9),
            nn.MaxPool1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size // 2, config.dis_emb_size // 4, kernel_size=9),
            nn.MaxPool1d(2),
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size // 4, config.dis_emb_size // 8, kernel_size=9),
        )
        self.rnn = nn.LSTM(
            input_size=config.dis_emb_size // 8, hidden_size=config.dis_emb_size // 16, bidirectional=True
        )
        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(config.dis_emb_size // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask):
        if mask is not None:
            x = x * mask.reshape(*x.shape[:2], 1)
        e = self.embedding(x)
        N, T, E = e.shape
        e = e.view(N, E, T)
        x = self.layers(e)   # (N, E', T)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)  # (N, T, E'')
        x = self.out(x)[:, :self.num_segment_map_param, 0] * T  # (N, phn_max_length)
        x, _ = x.sort()
        return x


# class SegmentMapGenerator(nn.Module):
#
#     def __init__(self, config, num_segment_map_param: int):
#         super().__init__()
#         self.embedding = nn.Linear(config.feat_dim, config.dis_emb_size, bias=False)
#         self.num_segment_map_param = num_segment_map_param
#
#         self.layers = nn.Sequential(
#             nn.LeakyReLU(),
#             nn.Conv1d(config.dis_emb_size, config.dis_emb_size // 2, kernel_size=9, padding=9 // 2),
#             nn.LeakyReLU(),
#             nn.Conv1d(config.dis_emb_size // 2, config.dis_emb_size // 4, kernel_size=9, padding=9 // 2),
#             nn.LeakyReLU(),
#             nn.Conv1d(config.dis_emb_size // 4, 2, kernel_size=9, padding=9 // 2),
#         )
#
#     def _forward(self, x: torch.Tensor, mask: torch.Tensor = None):
#         # x: (N, T, V)
#         if mask is not None:
#             x = x * mask.reshape(*x.shape[:2], 1)
#
#         e = self.embedding(x)
#         N, T, E = e.shape
#         e = e.view(N, E, T)
#         x = self.layers(e)  # (N, 2, T)
#         x = torch.cat([torch.tanh(x[:, 0:1, :]), torch.sigmoid(x[:, 1:2, :] - 3.)], dim=1) * T
#         return x
#
#     @staticmethod
#     def non_maximum_suppression(x: torch.Tensor, phn_max_length: int, mask: torch.Tensor = None):
#         if mask is None:
#             mask = torch.ones_like(x[:, 0, :]).to(x.device)
#
#         maxlen = x.shape[-1]
#         taken_mask = (1 - mask.long()).cpu()
#         selected_indices = torch.zeros([len(x), phn_max_length]).long()
#         selected_mask = torch.zeros_like(selected_indices).float()
#         cpu_x = x.cpu().data
#         seq_lens = torch.sum(mask.long(), dim=-1).cpu()
#         for i in range(len(x)):
#             seq_len = seq_lens[i]
#             displacements = cpu_x[i, 0]
#             distance_to_centers = displacements.abs()
#             distance_to_centers[seq_len:] = maxlen
#
#             for phn_idx in range(phn_max_length):
#                 if taken_mask[i].sum() == maxlen:
#                     break
#                 min_index = distance_to_centers.argmin().item()
#                 displacement = displacements[min_index].item()
#                 width = cpu_x[i, 1, min_index].item()
#
#                 selected_indices[i, phn_idx] = min_index
#                 selected_mask[i, phn_idx] = 1
#                 mask_start_index = int(np.clip(min_index + displacement - width / 2, a_min=0, a_max=maxlen - 1))
#                 mask_end_index = int(np.clip(min_index + displacement + width / 2, a_min=0, a_max=maxlen - 1))
#
#                 taken_mask[i, mask_start_index:mask_end_index + 1] = 1
#                 distance_to_centers[mask_start_index:mask_end_index + 1] = maxlen
#
#         selected_indices, selected_mask = selected_indices.to(x.device), selected_mask.to(x.device)
#         centers = selected_indices.float() + torch.gather(x[:, 0], dim=1, index=selected_indices)
#         centers = centers * selected_mask + (1 - selected_mask) * maxlen
#         return centers
#
#     def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
#         x = self._forward(x, mask)
#         x = self.non_maximum_suppression(x, self.num_segment_map_param, mask)
#         x, _ = x.sort(dim=-1)
#         return x


class SegmentMapper(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.phn_max_length = config.phn_max_length
        self.feat_max_length = config.feat_max_length

    def get_feat_indices(self, segment_map_params: torch.Tensor) -> torch.Tensor:
        # segment_map_params: (batch_size, phn_max_length)
        return segment_map_params

    @property
    def n_param(self) -> int:
        return self.phn_max_length

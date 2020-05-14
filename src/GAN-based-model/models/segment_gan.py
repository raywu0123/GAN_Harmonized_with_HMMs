import sys
import os
import math

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
        self.segment_map_generator = SegmentMapGenerator(config, self.config.phn_max_length)
        self.reconstruction_net = ReconstructionNet(config)
        self.critic = Critic(config)

        self.g_opt = torch.optim.Adam(
            params=list(self.generator.parameters()) + list(self.segment_map_generator.parameters()) + list(self.reconstruction_net.parameters()),
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
            self.reconstruction_net.cuda()
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
                    # self.train_critic(batch, get_target_batch, logger, config, frame_temp)
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

        feat_indices = self.segment_map_generator(batch_feats, feat_mask)
        selected_feats, mask = self.get_selected_feats(batch_feats, feat_lens, feat_indices)  # (batch_size, config.phn_max_length)
        fake_target_logits = self.generator(selected_feats)

        noisy_feat_indices = feat_indices + torch.empty_like(feat_indices).to(feat_indices.device).uniform_(-1, 1)
        noisy_selected_feats, _ = self.get_selected_feats(batch_feats, feat_lens, noisy_feat_indices)
        noisy_fake_target_logits = self.generator(noisy_selected_feats)
        return {
            "selected_feats": selected_feats,
            "selected_indices": feat_indices,
            "fake_target_logits": fake_target_logits,
            "noisy_fake_target_logits": noisy_fake_target_logits,
            "phn_mask": mask,
            'feats': batch_feats,
            "feat_mask": feat_mask,
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
        fake_target_logits, phn_mask = gen['fake_target_logits'], gen['phn_mask']
        fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
        fake_score = self.critic(fake_target_probs, phn_mask).mean()

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
        fake_target_logits, noisy_fake_target_logits, phn_mask = gen['fake_target_logits'], gen['noisy_fake_target_logits'], gen['phn_mask']

        # fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
        # fake_score = self.critic(fake_target_probs, phn_mask).mean()  # shape: (N, 1)

        # noisy_fake_target_probs = F.softmax(noisy_fake_target_logits / frame_temp, dim=-1)
        # segment_loss = ((fake_target_probs - noisy_fake_target_probs) ** 2).sum(-1).mean()

        # g_loss = -fake_score

        feats, selected_feats, selected_indices, feat_mask = gen['feats'], gen['selected_feats'], gen['selected_indices'], gen['feat_mask']
        recon_input = torch.cat([selected_feats, selected_indices.unsqueeze(2) / self.config.feat_max_length], dim=-1)  # (N, phn_max_length, feat_dim + 1)
        reconstruction = self.reconstruction_net(recon_input, phn_mask)
        recon_loss = masked_reduce_mean((reconstruction - feats) ** 2, feat_mask).mean()

        # total_loss = g_loss + self.config.seg_loss_ratio * segment_loss + self.config.recon_loss_ratio * recon_loss
        total_loss = recon_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.generator.parameters()) + list(self.segment_map_generator.parameters()) + list(self.reconstruction_net.parameters()),
            5.
        )
        self.g_opt.step()

        phn_mask = phn_mask.squeeze()  # (N, T)
        lengths = phn_mask.sum(dim=-1).cpu().data.numpy()  # (N,)
        true_lengths = np.array([len(bnd) for bnd in batch['source_bnd']], dtype=float)
        length_correlation = pearsonr(lengths, true_lengths)[0]

        # fake_sample = torch.argmax(fake_target_probs[0], dim=-1).long() * phn_mask[0].long()
        logger.update({
            # 'g_loss': g_loss.item(),
        }, group_name='GAN_losses')
        logger.update({
            # 'seg_loss': segment_loss.item(),
        }, group_name='segment_losses')
        logger.update({
            'recon_loss': recon_loss.item(),
            # 'fake_sample': array_to_string(fake_sample.cpu().data.numpy()),
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
        self.embedding = nn.Linear(config.feat_dim, config.dis_emb_size)
        self.num_segment_map_param = num_segment_map_param

        self.layers = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size, config.dis_emb_size, kernel_size=9, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size, config.dis_emb_size, kernel_size=9, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size, config.dis_emb_size, kernel_size=9, stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.dis_emb_size, config.dis_emb_size, kernel_size=9, stride=2),
        )
        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(config.dis_emb_size, self.num_segment_map_param),
        )

    def forward(self, x, mask):
        if mask is not None:
            x = x * mask.reshape(*x.shape[:2], 1)
        e = self.embedding(x)
        N, T, E = e.shape
        e = e.view(N, E, T)
        x = self.layers(e)   # (N, config.dis_emb_size, T')
        x = x.mean(dim=-1)  # (N, config.dis_emb_size)
        x = self.out(x)  # (N, phn_max_length)
        x = torch.sigmoid(x - 1) * T
        x, _ = x.sort(dim=-1)
        return x


class ReconstructionNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.feat_dim + 1, config.feat_dim, bias=False)
        self.encoder = nn.GRU(config.feat_dim, config.feat_dim, batch_first=True)
        self.decoder = nn.GRU(config.feat_dim, config.feat_dim, batch_first=True)

    def forward(self, x, mask=None):
        # X: (N, T, E)
        if mask is not None:
            x = x * mask.reshape(*x.shape[:2], 1)
        e = self.embedding(x)

        encoder_outputs, encoder_hiddens = self.encoder(e)

        decoder_inputs = e.repeat(1, math.ceil(self.config.feat_max_length / x.shape[1]), 1)[:, :self.config.feat_max_length]
        decoder_outputs, decoder_hiddens = self.decoder(decoder_inputs, encoder_outputs[:, -1].unsqueeze(0))
        return decoder_outputs

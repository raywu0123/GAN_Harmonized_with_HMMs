import sys
import os
import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    gumbel_sample,
    create_attention_mask,
    intra_segment_loss,
)
from lib.torch_gan_utils import (
    to_onehot,
    compute_gradient_penalty,
)
from lib.utils import array_to_string
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
        self.critic = Critic(config)

        self.g_opt = torch.optim.Adam(
            params=self.generator.parameters(),
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
        max_fer = 100.0
        frame_temp = 0.9
        for step in range(1, config.step + 1):
            if step == 8000:
                frame_temp = 0.8
            if step == 12000:
                frame_temp = 0.7

            self.generator.eval()
            for _ in range(config.dis_iter):
                self.c_opt.zero_grad()
                batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                feat_mask = create_attention_mask(batch_sample_len, config.phn_max_length).unsqueeze(2)

                real_target_idx, batch_target_len = get_target_batch(batch_size)
                batch_sample_feat = get_tensor_from_array(batch_sample_feat)
                real_target_idx = get_tensor_from_array(real_target_idx).long()
                real_target_probs = to_onehot(real_target_idx, class_num=config.phn_size)
                target_mask = create_attention_mask(batch_target_len, config.phn_max_length).unsqueeze(2)

                fake_target_logits = self.generator(batch_sample_feat)
                fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
                real_score = self.critic(real_target_probs, target_mask).mean()
                fake_score = self.critic(fake_target_probs, feat_mask).mean()

                real_target_probs = (real_target_probs * target_mask)
                fake_target_probs = (fake_target_probs * feat_mask)
                inter_alphas = torch.empty(
                    [fake_target_probs.shape[0], 1, 1],
                    device=fake_target_probs.device,
                ).uniform_()
                inter_samples = real_target_probs + inter_alphas * (fake_target_probs.detach() - real_target_probs)
                inter_samples = torch.tensor(inter_samples, requires_grad=True)
                inter_score = self.critic(inter_samples).mean()
                gradient_penalty = compute_gradient_penalty(inter_score, inter_samples)

                c_loss = -real_score + fake_score + config.penalty_ratio * gradient_penalty
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.)
                self.c_opt.step()
                logger.update({
                    'c_loss': c_loss.item(),
                    'true_sample': array_to_string(real_target_idx[0].cpu().data.numpy()),
                    'gradient_penalty': gradient_penalty.item(),
                }, group_name='GAN_losses')

            self.generator.train()
            self.critic.eval()
            for _ in range(config.gen_iter):
                self.g_opt.zero_grad()
                batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                batch_sample_feat = get_tensor_from_array(batch_sample_feat)
                mask = create_attention_mask(batch_sample_len, config.phn_max_length).unsqueeze(2)
                batch_repeat_num = get_tensor_from_array(batch_repeat_num)

                fake_target_logits = self.generator(batch_sample_feat)
                fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
                fake_score = self.critic(fake_target_probs, mask).mean()  # shape: (N, 1)
                g_loss = -fake_score

                segment_loss = intra_segment_loss(
                    fake_target_logits,
                    batch_repeat_num,
                    mask.squeeze(),
                    sep_size=(config.batch_size * config.repeat) // 2,
                )
                total_loss = g_loss + config.seg_loss_ratio * segment_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5.)
                self.g_opt.step()

                fake_sample = torch.argmax(fake_target_probs[0], dim=-1) * mask[0].squeeze().long()
                logger.update({
                    'g_loss': g_loss.item(),
                    'fake_sample': array_to_string(fake_sample.cpu().data.numpy()),
                }, group_name='GAN_losses')
                logger.update({'seg_loss': segment_loss.item()}, group_name='segment_losses')

            self.critic.train()
            if step % config.eval_step == 0:
                step_fer = frame_eval(self.predict_batch, dev_data_loader)
                logger.update({'fer': step_fer}, ema=False, group_name='val')
                print(f'EVAL max: {max_fer:.2f} step: {step_fer:.2f}')
                if step_fer < max_fer:
                    max_fer = step_fer
            logger.step()

        print('=' * 80)

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

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        self.generator.eval()
        with torch.no_grad():
            batch_frame_feat = get_tensor_from_array(batch_frame_feat)
            batch_frame_logits = self.generator(batch_frame_feat)
            batch_frame_prob = torch.softmax(batch_frame_logits, dim=-1)

        batch_frame_prob = batch_frame_prob.cpu().data.numpy()
        self.generator.train()
        return batch_frame_prob


class Generator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.feat_dim, config.gen_hidden_size)
        self.dense_out = nn.Linear(config.gen_hidden_size, config.phn_size)

    def forward(self, x) -> torch.Tensor:
        x = self.dense_in(x)
        x = torch.relu(x)
        logits = self.dense_out(x)
        return logits


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

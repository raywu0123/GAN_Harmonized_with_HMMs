import sys
import os
import math

import torch
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    gumbel_sample,
    epsilon,
    EMA,
    create_attention_mask,
    masked_reduce_sum,
    first_order_expand,
    intra_segment_loss,
)
from lib.utils import array_to_string
from evalution import frame_eval
from callbacks import Logger


class SwapGANModel(ModelBase):
    description = "SWAPGAN MODEL"

    def __init__(self, config, wgan=True):
        self.config = config
        self.wgan = wgan

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.generator = Generator(config)
        self.critic = Critic(config, wgan=wgan)

        self.g_opt = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=config.gen_lr,
            betas=[0.5, 0.999],
        )
        self.c_opt = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=config.dis_lr,
            betas=[0.5, 0.999],
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

        for step in range(1, config.step + 1):

            self.generator.eval()
            for _ in range(config.dis_iter):
                self.c_opt.zero_grad()
                batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                real_target_idx, batch_target_len = get_target_batch(batch_size)
                batch_sample_feat = get_tensor_from_array(batch_sample_feat)
                real_target_idx = get_tensor_from_array(real_target_idx).long()
                mask = create_attention_mask(batch_sample_len, config.phn_max_length)

                fake_target_logits, fake_target_idx = self.generator(batch_sample_feat, mask)

                real_score = self.critic(real_target_idx)
                fake_score = self.critic(fake_target_idx)

                if not self.wgan:
                    c_loss = torch.mean(-torch.log(real_score + epsilon)) + \
                        torch.mean(-torch.log(1 - fake_score + epsilon))
                else:
                    c_loss = torch.mean(-real_score) + torch.mean(fake_score)
                c_loss.backward()
                self.c_opt.step()
                logger.update({
                    'c_loss': c_loss.item(),
                    'true_sample': array_to_string(real_target_idx[0].cpu().data.numpy()),
                })

            self.generator.train()
            self.critic.eval()
            for _ in range(config.gen_iter):
                self.g_opt.zero_grad()
                batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                batch_sample_feat = get_tensor_from_array(batch_sample_feat)
                mask = create_attention_mask(batch_sample_len, config.phn_max_length)
                batch_repeat_num = get_tensor_from_array(batch_repeat_num)

                fake_target_logits, fake_target_idx = self.generator(batch_sample_feat, mask)
                fake_score = self.critic(fake_target_idx)  # shape: (N, 1)
                reward = self.critic.compute_G_reward(fake_score)
                kernel = self.critic.get_kernel()
                g_loss = self.generator.compute_loss(reward, kernel, fake_target_logits, fake_target_idx, mask)
                segment_loss = intra_segment_loss(
                    fake_target_logits,
                    batch_repeat_num,
                    mask,
                    sep_size=(config.batch_size * config.repeat) // 2,
                )
                total_loss = g_loss + config.seg_loss_ratio * segment_loss
                total_loss.backward()

                self.g_opt.step()
                logger.update({
                    'g_loss': g_loss.item(),
                    'seg_loss': segment_loss.item(),
                    'fake_sample': array_to_string(fake_target_idx[0].cpu().data.numpy()),
                    'baseline': self.critic.ema.average.item(),
                })

            self.critic.train()
            if step % config.eval_step == 0:
                step_fer = frame_eval(self.predict_batch, dev_data_loader)
                logger.update({'val_fer': step_fer}, ema=False)
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
            mask = create_attention_mask(batch_frame_len, batch_frame_feat.shape[1])
            batch_frame_logits, _ = self.generator(batch_frame_feat, mask)
            batch_frame_prob = torch.softmax(batch_frame_logits, dim=-1)

        batch_frame_prob = batch_frame_prob.cpu().data.numpy()
        self.generator.train()
        return batch_frame_prob


class Generator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.feat_dim, config.gen_hidden_size)
        self.dense_out = nn.Linear(config.gen_hidden_size, config.phn_size)

    def forward(self, x, mask):
        x = self.dense_in(x)
        x = torch.relu(x)
        logits = self.dense_out(x)
        logits = logits * mask.unsqueeze(2)
        idx = gumbel_sample(logits) * mask.long()  # shape: (N, T)
        return logits, idx

    def compute_loss(self, reward, kernel, target_logits, target_idx, mask, normalize_type=0):
        target_probs = torch.softmax(target_logits, dim=-1)  # shape: (N, T, V)
        batch_kernel = F.embedding(input=target_idx, weight=kernel)  # shape: (N, T, V)
        if normalize_type == 1:
            selected_probs = torch.gather(
                target_probs,
                dim=-1,
                index=target_idx.unsqueeze(2),
            )  # shape: (N, T, 1)
            batch_kernel /= (selected_probs + epsilon)
        else:
            batch_kernel /= (torch.einsum('ntv,vq->ntq', target_probs, kernel) + epsilon)
        full_loss = -(batch_kernel * reward).detach() * target_probs
        seq_loss = torch.sum(full_loss, dim=-1)  # shape: (N, T)
        sample_loss = masked_reduce_sum(seq_loss, mask)  # shape: (N,)
        loss = torch.mean(sample_loss)
        return loss


class Critic(nn.Module):

    def __init__(self, config, kernel_sizes=(3, 5, 7, 9), wgan=False):
        super().__init__()
        self.wgan = wgan
        self.kernel_sizes = kernel_sizes
        self.ema = EMA(0.9)
        self.embedding = nn.Embedding(config.phn_size, embedding_dim=config.dis_emb_size)
        self.e = None
        self.first_channels = config.dis_emb_size
        self.first_convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv1d(
                config.dis_emb_size,
                self.first_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ))
            for kernel_size in kernel_sizes
        ])
        self.second_channels = config.dis_emb_size
        self.second_convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv1d(
                self.first_channels * len(kernel_sizes),
                self.second_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ))
            for kernel_size in kernel_sizes
        ])
        self.activation = torch.nn.LeakyReLU()
        self.dense = nn.utils.spectral_norm(
            nn.Linear(config.phn_max_length * self.second_channels * len(kernel_sizes), 1)
        )

    def forward(self, x):
        # x: (N, T)
        assert(x.ndimension() == 2)
        e = self.embedding(x)  # shape: (N, T, E)
        self.e = e
        N, T, E = e.shape

        e = e.view(N, E, T)
        outputs = torch.cat([
            self.activation(conv(e)) / math.sqrt(conv.kernel_size[0]) for conv in self.first_convs
        ], dim=1)
        outputs = outputs / math.sqrt(len(self.kernel_sizes))
        outputs = torch.cat([
            self.activation(conv(outputs)) / math.sqrt(conv.kernel_size[0]) for conv in self.second_convs
        ], dim=1)
        outputs = outputs / math.sqrt(len(self.kernel_sizes))

        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.dense(outputs)
        if not self.wgan:
            return torch.sigmoid(outputs)
        else:
            return outputs

    def get_emb_vecs(self):
        assert(self.e is not None)
        return self.e

    def get_embedding_weights(self):
        return self.embedding.weight  # shape: (V, E)

    def get_kernel(self):
        emb_vecs = self.get_embedding_weights()
        V, E = emb_vecs.shape
        emb_vecs_square = torch.sum(emb_vecs ** 2, dim=-1)
        x_square = emb_vecs_square.view(V, 1)
        y_square = emb_vecs_square.view(1, V)
        xy_term = torch.einsum('ae,be->ab', emb_vecs, emb_vecs)  # shape: (V, V)
        square_dist = -2 * xy_term + x_square + y_square
        return torch.softmax(-2. * square_dist, dim=0)

    def compute_G_reward(self, fake_score):
        if not self.wgan:
            reward = torch.log(fake_score + epsilon)
        else:
            reward = fake_score
        mean_reward = torch.mean(reward)
        emb_vecs = self.get_emb_vecs()  # shape: (N, T, E)
        N = reward.shape[0]
        grad_on_emb = grad(outputs=N * mean_reward, inputs=emb_vecs)[0]  # shape: (N, T, E)
        embedding_weights = self.get_embedding_weights()  # shape: (V, E)
        first_order_reward = first_order_expand(grad_on_emb, emb_vecs, embedding_weights)
        ema_reward = self.ema(mean_reward)
        advantage = first_order_reward + (reward - ema_reward).view(-1, 1, 1)
        return advantage

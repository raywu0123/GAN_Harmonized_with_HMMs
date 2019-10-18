import sys
import os

import torch
from torch import nn
import numpy as np

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    gumbel_sample,
    epsilon,
)
from evalution import phn_eval


class SwapGANModel(ModelBase):

    description = "SWAPGAN MODEL"

    def __init__(self, config):
        self.config = config
        self.align_layer_idx = -1

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.generator = Generator(config)
        self.critic = Critic(config)

        self.g_opt = torch.optim.Adam(params=self.generator.parameters())
        self.c_opt = torch.optim.Adam(params=self.critic.parameters())

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
        max_err = 100.0

        for step in range(1, config.step + 1):
            for _ in range(config.dis_iter):
                self.c_opt.zero_grad()
                batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                real_target_idx, batch_target_len = get_target_batch(batch_size)

                fake_target_logits = self.generator(batch_sample_feat)
                fake_target_idx = gumbel_sample(fake_target_logits)  # shape: (N, T, 1)

                real_score = self.critic(real_target_idx)
                fake_score = self.critic(fake_target_idx)

                c_loss = torch.mean(torch.log(real_score + epsilon) + torch.log(1 - fake_score + epsilon))
                c_loss.backward()
                self.c_opt.step()

            for _ in range(config.gen_iter):
                self.g_opt.zero_grad()
                batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                    config.batch_size,
                    repeat=config.repeat,
                )
                fake_target_logits = self.generator(batch_sample_feat)
                fake_target_idx = gumbel_sample(fake_target_logits)  # shape: (N, T, 1)
                fake_score = self.critic(fake_target_idx)
                g_loss = torch.mean(torch.log(fake_score + epsilon))
                g_loss.backward()

                grad_on_emb = self.critic.get_grad_on_emb()  # shape: (N, T, E)
                fake_target_probs = torch.softmax(fake_target_logits, dim=-1)
                intermediate_loss = torch.mean(grad_on_emb * fake_target_probs)
                intermediate_loss.backward()
                self.g_opt.step()

            if step % config.eval_step == 0:
                pass
                # TODO

        print('=' * 80)

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(
            {
                'state_dict': self.bert_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            os.path.join(save_path, 'checkpoint.pth.tar')
        )

    def restore(self, save_dir):
        # TODO
        pass

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        with torch.no_grad():
            pass
            # TODO


class Generator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config.feat_dim, config.gen_hidden_size)
        self.dense_2 = nn.Linear(config.gen_hidden_size, config.phn_size)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dense_2(x)
        logits = x
        return logits


class Critic(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.phn_size, embedding_dim=config.dis_emb_size)
        self.e = None

    def forward(self, x):
        e = self.embedding(x)
        self.e = e
        score = torch.sigmoid(torch.mean(torch.mean(e, dim=2), dim=1))
        return score

    def get_grad_on_emb(self):
        assert(self.e is not None)
        return self.e.detach()

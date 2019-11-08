import sys
import os

import torch
from torch import nn
from torch.nn import functional as F

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    cpc_loss,
    masked_reduce_mean,
)
from lib.torch_gan_utils import (
    to_onehot,
    compute_gradient_penalty,
)
from lib.torch_bert_utils import get_attention_mask
from lib.utils import array_to_string
from evalution import phn_eval
from callbacks import Logger


class SegmentGANModel(ModelBase):

    description = "SEGMENT-GAN MODEL"

    def __init__(self, config):
        self.config = config

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        self.generator = Generator(config)
        self.reverse_generator = ReverseGenerator(config)
        self.critic = Critic(config)

        self.g_opt = torch.optim.Adam(
            params=list(self.generator.parameters()) + list(self.reverse_generator.parameters()),
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
            self.reverse_generator.cuda()
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
                    self.train_generator(batch, logger, config, frame_temp, get_target_batch)

                if step % config.eval_step == 0:
                    step_err, labels, preds = phn_eval(
                        self.predict_batch,
                        dev_data_loader,
                        batch_size=batch_size,
                    )
                    print(f'EVAL max: {max_err:.2f} step: {step_err:.2f}')
                    logger.update({'per': step_err}, ema=False, group_name='val')
                    logger.update({
                        "LABEL  ": " ".join(["%3s" % str(l) for l in labels[0]]),
                        "PREDICT": " ".join(["%3s" % str(p) for p in preds[0]]),
                    }, ema=False)
                    if step_err < max_err:
                        max_err = step_err
                        self.save(config.save_path)
                logger.step()
                step += 1
        print('=' * 80)

    def train_critic(self, batch, get_target_batch, logger, config, frame_temp):
        self.c_opt.zero_grad()
        feats, feat_lens = batch['source'], batch['source_length']
        batch_feats = get_tensor_from_array(feats)
        batch_size = len(feats)
        real_target_idx, batch_target_len = get_target_batch(batch_size)
        real_target_idx = get_tensor_from_array(real_target_idx).long()
        real_target_probs = to_onehot(real_target_idx, class_num=config.phn_size)

        fake_target_logits = self.generator(batch_feats).detach()
        fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
        real_score = self.critic(real_target_probs).mean()
        fake_score = self.critic(fake_target_probs).mean()

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
            'true_sample': array_to_string(real_target_idx[0].cpu().data.numpy()),
            'gradient_penalty': gradient_penalty.item(),
        }, group_name='GAN_losses')

    def train_generator(self, batch, logger, config, frame_temp, get_target_batch):
        self.g_opt.zero_grad()
        feats, feat_lens = batch['source'], batch['source_length']
        batch_feats = get_tensor_from_array(feats)
        mask = get_attention_mask(feat_lens, max_len=config.feat_max_length)
        fake_target_logits = self.generator(batch_feats)

        fake_target_probs = F.softmax(fake_target_logits / frame_temp, dim=-1)
        fake_score = self.critic(fake_target_probs).mean()  # shape: (N, 1)

        reverse_feats = self.reverse_generator(fake_target_probs)
        feat_recon_loss = cpc_loss(reverse_feats, batch_feats, mask, mask)
        real_target_idx, batch_target_len = get_target_batch(batch_size=len(feats))
        target_mask = get_attention_mask(batch_target_len, max_len=config.phn_max_length)

        real_target_idx = get_tensor_from_array(real_target_idx).long()
        real_target_probs = to_onehot(real_target_idx, class_num=config.phn_size)
        recon_target_logits = self.generator(self.reverse_generator(real_target_probs).detach()).transpose(1, 2)
        target_recon_loss = nn.CrossEntropyLoss(reduction='none')(recon_target_logits, real_target_idx)
        target_recon_loss = masked_reduce_mean(target_recon_loss, target_mask).mean()

        g_loss = -fake_score + feat_recon_loss + target_recon_loss
        total_loss = g_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.generator.parameters()) + list(self.reverse_generator.parameters()),
            5.
        )
        self.g_opt.step()

        fake_sample = torch.argmax(fake_target_probs[0], dim=-1).long()
        logger.update({
            'g_loss': g_loss.item(),
            'fake_sample': array_to_string(fake_sample.cpu().data.numpy()),
        }, group_name='GAN_losses')
        logger.update({
            'feat_recon_loss': feat_recon_loss.item(),
            'target_recon_loss': target_recon_loss.item(),
        }, group_name='recon_losses')

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

    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.feat_dim, config.gen_hidden_size)
        self.dense_out = nn.Linear(config.gen_hidden_size, config.phn_size)
        self.cnn = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=config.gen_hidden_size, out_channels=config.gen_hidden_size, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=config.gen_hidden_size, out_channels=config.gen_hidden_size, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=config.gen_hidden_size, out_channels=config.gen_hidden_size, kernel_size=9, stride=2),
        )
        self.phn_max_length = config.phn_max_length

    def forward(self, x, mask=None) -> torch.Tensor:
        x = self.dense_in(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = F.interpolate(x, [self.phn_max_length])
        x = x.transpose(1, 2)
        x = F.relu(x)
        logits = self.dense_out(x)
        return logits


class ReverseGenerator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.phn_size, config.gen_hidden_size)
        self.dense_out = nn.Linear(config.gen_hidden_size, config.feat_dim)
        self.feat_max_length = config.feat_max_length
        self.dcnn = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=config.gen_hidden_size, out_channels=config.gen_hidden_size, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=config.gen_hidden_size, out_channels=config.gen_hidden_size, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=config.gen_hidden_size, out_channels=config.gen_hidden_size, kernel_size=9, stride=2)
        )

    def forward(self, x):
        x = self.dense_in(x)
        x = x.transpose(1, 2)
        x = self.dcnn(x)
        x = F.interpolate(x, size=[self.feat_max_length], mode='linear')
        x = x.transpose(1, 2)
        x = F.relu(x)
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

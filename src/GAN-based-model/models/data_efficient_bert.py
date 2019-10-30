import sys
import os
import copy

import torch
from torch import nn
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel,
    BertConfig,
    BertLayer,
)
from pytorch_pretrained_bert.optimization import BertAdam

from evalution import frame_eval
from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    cpc_loss,
    masked_reduce_mean,
)
from lib.torch_bert_utils import (
    get_attention_mask,
    get_mlm_masks,
    PositionalEncoding,
)
from callbacks import Logger


class DataEfficientBert(ModelBase):

    description = "DATA EFFICIENT BERT MODEL"

    def __init__(self, config):
        self.config = config
        self.align_layer_idx = -1

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        bert_config = BertConfig(
            vocab_size_or_config_json_file=config.phn_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        self.bert_model = BertModel(
            bert_config,
            config.feat_dim,
            config.phn_size,
            (config.batch_size * config.repeat) // 2,
        )
        self.optimizer = BertAdam(
            params=self.bert_model.parameters(),
            lr=config.sup_lr,
            warmup=0.1,
            t_total=config.pretrain_step + config.finetune_step,
        )
        if torch.cuda.is_available():
            self.bert_model.cuda()

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
        logger = Logger(config.print_step)
        print(f'Pretraining for {config.pretrain_step} steps...')
        self.pretrain(
            data_loader=data_loader,
            dev_data_loader=dev_data_loader,
            batch_size=batch_size,
            config=config,
            logger=logger
        )
        print(f'Finetuning for {config.finetune_step} steps...')
        data_loader.set_use_ratio(use_ratio=config.finetune_ratio, verbose=True)
        self.finetune(
            data_loader=data_loader,
            dev_data_loader=dev_data_loader,
            batch_size=batch_size,
            config=config,
            logger=logger,
            aug=aug,
        )
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
        checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        self.bert_model = checkpoint['state_dict']
        self.optimizer = checkpoint['optimizer']

    def pretrain(self, data_loader, dev_data_loader, batch_size, config, logger):
        i_step = 0
        for i_epoch in range(1, config.epoch + 1):
            for batch in data_loader.get_batch(batch_size):
                batch_frame_feat, batch_frame_len = batch['source'], batch['source_length']
                self.optimizer.zero_grad()
                loss = self.bert_model.pretrain_loss(batch_frame_feat, batch_frame_len)
                loss.backward()
                self.optimizer.step()
                logger.update({'train_loss': loss.item()}, group_name='pretrain')
                if i_step % config.eval_step == 0:
                    self.pretrain_eval(dev_data_loader, batch_size, logger)
                logger.step()
                i_step += 1
                if i_step == config.pretrain_step:
                    return

    def pretrain_eval(self, dev_data_loader, batch_size, logger):
        self.bert_model.eval()
        with torch.no_grad():
            for batch in dev_data_loader.get_batch(batch_size):
                batch_frame_feat, batch_frame_len = batch['source'], batch['source_length']
                loss = self.bert_model.pretrain_loss(batch_frame_feat, batch_frame_len)
                logger.update({'val_loss': loss.item()}, group_name='pretrain')
        self.bert_model.train()

    def finetune(self, data_loader, dev_data_loader, batch_size, config, logger, aug):
        i_step = 1
        for i_epoch in range(1, config.epoch + 1):
            for batch in data_loader.get_batch(batch_size):
                batch_frame_feat, batch_frame_label, batch_frame_len = \
                    batch['source'], batch['frame_label'], batch['source_length']
                self.optimizer.zero_grad()
                loss = self.bert_model.finetune_loss(batch_frame_feat, batch_frame_label, batch_frame_len)
                loss.backward()
                self.optimizer.step()
                logger.update({'train_loss': loss.item()}, group_name='finetune')
                if i_step % config.eval_step == 0:
                    self.finetune_eval(dev_data_loader, batch_size, logger)
                logger.step()
                i_step += 1
                if i_step == config.finetune_step:
                    return

    def finetune_eval(self, dev_data_loader, batch_size, logger):
        step_fer = frame_eval(
            self.predict_batch,
            dev_data_loader,
            batch_size=batch_size,
        )
        logger.update({'val_fer': step_fer}, ema=False)

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        self.bert_model.eval()
        with torch.no_grad():
            pred = self.bert_model.predict(batch_frame_feat, batch_frame_len)
            pred = pred.cpu().data.numpy()
        self.bert_model.train()
        return pred


class BertModel(BertPreTrainedModel):

    def __init__(
        self,
        config,
        feat_dim,
        phn_size,
        sep_size,
        mask_prob=0.15,
        mask_but_no_prob=0.1,
    ):
        super(BertModel, self).__init__(config)
        self.mask_prob = mask_prob
        self.mask_but_no_prob = mask_but_no_prob
        self.sep_size = sep_size

        self.feat_embeddings = nn.Linear(feat_dim, config.hidden_size)
        self.feat_mask_vec = nn.Parameter(torch.zeros(feat_dim), requires_grad=True)
        self.positional_encoding = PositionalEncoding(config.hidden_size)

        self.model = BertModel
        layer = BertLayer(config)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.feat_out_layer = nn.Linear(config.hidden_size, feat_dim)
        self.target_out_layer = nn.Linear(config.hidden_size, phn_size)
        self.apply(self.init_bert_weights)

    def pretrain_loss(self, input_feats, seq_lens):
        input_feats = get_tensor_from_array(input_feats)
        attention_mask = get_attention_mask(seq_lens, input_feats.shape[1])
        input_mask, predict_mask = get_mlm_masks(input_feats, self.mask_prob, self.mask_but_no_prob)

        masked_input_feats = input_mask * input_feats + (1 - input_mask) * self.feat_mask_vec
        masked_input_feats *= attention_mask.unsqueeze(2)  # taking care of the paddings

        x = self.feat_embeddings(masked_input_feats)
        x = self.positional_encoding(x)
        output = self.forward(x, attention_mask)
        output = self.feat_out_layer(output)

        to_predict = (1 - predict_mask.squeeze()) * attention_mask  # shape: (N, T)
        loss = cpc_loss(output, input_feats, to_predict, attention_mask)
        return loss

    def finetune_loss(self, frame_feat, frame_label, lens):
        outputs = self.predict(frame_feat, lens)
        outputs = outputs.transpose(1, 2)
        frame_label = get_tensor_from_array(frame_label).long()
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, frame_label)
        mask = get_attention_mask(lens, frame_feat.shape[1])
        loss = masked_reduce_mean(loss, mask)
        loss = loss.mean()
        return loss

    def predict(self, frame_feat, lens):
        frame_feat = get_tensor_from_array(frame_feat)
        mask = get_attention_mask(lens, frame_feat.shape[1])
        x = self.feat_embeddings(frame_feat)
        x = self.positional_encoding(x)
        outputs = self.forward(x, mask)
        outputs = self.target_out_layer(outputs)
        return outputs

    def forward(self, embedding_output, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(embedding_output[:, :, 0])

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        x = embedding_output
        for layer in self.encoder:
            x = layer(x, extended_attention_mask)
        return x

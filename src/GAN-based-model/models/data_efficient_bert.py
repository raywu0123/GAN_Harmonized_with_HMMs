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

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    cpc_loss,
)
from lib.torch_bert_utils import (
    get_attention_mask,
    get_mlm_masks,
)


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
            lr=3e-5,
            warmup=0.1,
            t_total=config.step,
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

        print('Pretraining...')
        self.pretrain(
            data_loader=data_loader,
            dev_data_loader=dev_data_loader,
            batch_size=batch_size,
            config=config,
        )
        print('Finetuning...')
        self.finetune(
            data_loader=data_loader,
            dev_data_loader=dev_data_loader,
            batch_size=batch_size,
            config=config,
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

    def predict_batch(self, batch_frame_feat, batch_frame_len):
        self.bert_model.eval()
        with torch.no_grad():
            predict_target_logits = self.bert_model.predict_targets_from_feats(
                batch_frame_feat,
                batch_frame_len,
            )
            frame_prob = torch.softmax(predict_target_logits, dim=-1)
            frame_prob = frame_prob.cpu().data.numpy()
        self.bert_model.train()
        return frame_prob

    def pretrain(self, data_loader, dev_data_loader, batch_size, config):
        for i_step in range(1, config.step + 1):
            batch = data_loader.get_batch(batch_size)
            batch_frame_feat, batch_frame_len = batch['source'], batch['source_length']
            self.optimizer.zero_grad()
            loss = self.bert_model.pretrain_loss(batch_frame_feat, batch_frame_len)
            loss.backward()
            self.optimizer.step()

    def finetune(self, data_loader, dev_data_loader, batch_size, config, aug):
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

    def phn_eval(self, data_loader, batch_size, repeat):
        pass


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
        self.feat_mask_vec = nn.Parameter(torch.zeros(feat_dim))

        self.target_embeddings = nn.Embedding(phn_size + 1, config.hidden_size)
        # + 1 for [MASK] token
        self.mask_token = phn_size

        layer = BertLayer(config)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.dropout = nn.Dropout(p=0.5)
        self.feat_out_layer = nn.Linear(config.hidden_size, feat_dim)
        self.target_out_layer = nn.Linear(config.hidden_size, phn_size)
        self.apply(self.init_bert_weights)

    def pretrain_loss(self, input_feats, seq_lens):
        input_feats = get_tensor_from_array(input_feats)
        attention_mask = get_attention_mask(seq_lens, input_feats.shape[1])
        input_mask, predict_mask = get_mlm_masks(input_feats, self.mask_prob, self.mask_but_no_prob)

        masked_input_feats = input_mask * input_feats + (1 - input_mask) * self.feat_mask_vec
        masked_input_feats *= attention_mask.unsqueeze(2)  # taking care of the paddings

        embedding_output = self.feat_embeddings(masked_input_feats)
        output = self.forward(embedding_output, attention_mask)
        output = self.feat_out_layer(output)

        to_predict = (1 - predict_mask.squeeze()) * attention_mask  # shape: (N, T)
        loss = cpc_loss(output, input_feats, to_predict, attention_mask)
        return loss

    def forward(self, embedding_output, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(embedding_output[:, :, 0])

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        x = embedding_output
        for layer in self.encoder:
            x = layer(x, extended_attention_mask)
            x = self.dropout(x)
        return x

import sys
import os

import torch
from torch import nn
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
    BertEmbeddings,
)
from pytorch_pretrained_bert.optimization import BertAdam
from callbacks import Logger
import numpy as np

from .base import ModelBase
from lib.torch_utils import (
    get_tensor_from_array,
    masked_reduce_mean,
    cpc_loss,
    intra_segment_loss,
    inter_segment_loss,
)
from lib.torch_bert_utils import (
    get_mlm_masks,
    get_attention_mask,
)
from evalution import phn_eval
use_cuda = torch.cuda.is_available() and False

class UnsBertModel(ModelBase):

    description = "UNSUPERVISED BERT MODEL"

    def __init__(self, config):
        self.config = config
        self.align_layer_idx = -1

        cout_word = f'{self.description}: building    '
        sys.stdout.write(cout_word)
        sys.stdout.flush()

        bert_config = BertConfig(
            vocab_size_or_config_json_file=config.phn_size,
            attention_probs_dropout_prob=0.1,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            type_vocab_size=2
        )
        self.pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model = MyBertModel(
            bert_config,
            config.feat_dim,
            config.phn_size,
            (config.batch_size * config.repeat) // 2,
            pretrained_embeddings=self.pretrained_bert.embeddings,
            pretrained_encoder=self.pretrained_bert.encoder,
        )
        self.optimizer = BertAdam(
            params=self.bert_model.parameters(),
            lr=3e-5,
            warmup=0.1,
            t_total=config.step,
        )
        if use_cuda:
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
        if aug:
            get_target_batch = data_loader.get_aug_target_batch
        else:
            get_target_batch = data_loader.get_target_batch

        logger = Logger(config.print_step)
        batch_size = config.batch_size * config.repeat
        max_err = 100.0

        for step in range(1, config.step + 1):
            batch_sample_feat, batch_sample_len, batch_repeat_num, batch_phn_label = data_loader.get_sample_batch(
                config.batch_size,
                repeat=config.repeat,
            )
            self.optimizer.zero_grad()
            feat_loss, intra_s_loss, inter_s_loss = self.bert_model.predict_feats(batch_sample_feat, batch_sample_len, batch_repeat_num)
            batch_target_idx, batch_target_len = get_target_batch(batch_size)
            target_loss = self.bert_model.predict_targets(batch_target_idx, batch_target_len)

            total_loss = feat_loss + target_loss + intra_s_loss + inter_s_loss
            total_loss.backward()
            self.optimizer.step()

            logger.update({
                'feat_loss': feat_loss.item(),
                'intra_s_loss': intra_s_loss.item(),
                'inter_s_loss': inter_s_loss.item(),
                'target_loss': target_loss.item(),
                'total_loss': total_loss.item(),
            })
            if step % config.eval_step == 0:
                step_err, labels, preds = self.phn_eval(
                    data_loader,
                    batch_size=batch_size,
                    repeat=config.repeat,
                )
                print(f'EVAL max: {max_err:.2f} step: {step_err:.2f}')
                logger.update({'val_per': step_err}, ema=False)
                logger.update({
                    "LABEL": " ".join(["%3s" % str(l) for l in labels[0]]),
                    "PREDICT": " ".join(["%3s" % str(p) for p in preds[0]]),
                }, ema=False)
                if step_err < max_err:
                    max_err = step_err
                    self.save(config.save_path)

            logger.step()

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

    def phn_eval(self, data_loader, batch_size, repeat):
        # self.bert_model.eval()
        with torch.no_grad():
            error_counts = []
            lens = []
            for _ in range(10):
                batch_feat, batch_feat_len, _, batch_phn_label = data_loader.get_sample_batch(
                    batch_size,
                    repeat=repeat,
                )
                batch_target_idx, batch_target_len = data_loader.get_aug_target_batch(batch_size)
                batch_prob = self.bert_model.predict_targets_from_feats(
                    batch_feat, batch_feat_len,
                    batch_target_idx, batch_target_len,
                )
                batch_prob = batch_prob.cpu().data.numpy()
                batch_pred = np.argmax(batch_prob, axis=-1)
                error_count, label_length, sample_labels, sample_preds = phn_eval(batch_pred, batch_feat_len, batch_phn_label, data_loader.phn_mapping)
                error_counts.append(error_count)
                lens.append(label_length)

        # self.bert_model.train()
        err = np.sum(error_counts) / np.sum(lens) * 100
        return err, sample_labels, sample_preds


class MyBertModel(BertPreTrainedModel):

    def __init__(
        self,
        config,
        feat_dim,
        phn_size,
        sep_size,
        pretrained_embeddings,
        pretrained_encoder,
        mask_prob=0.15,
        mask_but_no_prob=0.1,
        translate_layer_idx=-1,
        update_ratio=0.99,
    ):
        super(MyBertModel, self).__init__(config)
        self.mask_prob = mask_prob
        self.mask_but_no_prob = mask_but_no_prob
        self.translate_layer_idx = translate_layer_idx
        self.update_ratio = update_ratio
        self.sep_size = sep_size

        self.translation_dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.shadow_feat_inner = torch.zeros(config.hidden_size)
        self.shadow_target_inner = torch.zeros(config.hidden_size)
        if use_cuda:
            self.shadow_feat_inner = self.shadow_feat_inner.cuda()
            self.shadow_target_inner = self.shadow_target_inner.cuda()

        self.target_embeddings = nn.Linear(phn_size, config.hidden_size)
        self.feat_embeddings = nn.Linear(feat_dim, config.hidden_size)
        self.feat_out_layer = nn.Linear(config.hidden_size, feat_dim)
        self.target_out_layer = nn.Linear(config.hidden_size, phn_size)
        self.apply(self.init_bert_weights)

        self.pretrained_embeddings = pretrained_embeddings
        self.encoder = pretrained_encoder
        self.cls_token, self.sep_token, self.mask_token = 101, 102, 103

    def predict_feats(self, feats, seq_lens, repeats):
        feats, embedding_output, attention_mask, predict_mask = self.embed_feats(feats, seq_lens, mask_lm=True)

        feat_inner = self.forward(embedding_output, attention_mask, end_layer_idx=self.translate_layer_idx)
        output = self.forward(feat_inner, attention_mask, start_layer_idx=self.translate_layer_idx)
        output = self.feat_out_layer(output)

        to_predict = (1 - predict_mask.squeeze()) * attention_mask  # shape: (N, T)
        loss = cpc_loss(output, feats, to_predict, attention_mask)

        translation_inner = self.translate(feat_inner)
        translated_logits = self.forward(translation_inner, attention_mask, start_layer_idx=self.translate_layer_idx)
        translated_logits = self.target_out_layer(translated_logits)

        repeats = get_tensor_from_array(repeats)
        intra_s_loss = intra_segment_loss(translated_logits, repeats, attention_mask, self.sep_size)
        inter_s_loss = inter_segment_loss(translated_logits, attention_mask)

        self.update_shadow_variable(self.shadow_feat_inner, feat_inner, attention_mask)
        return loss, intra_s_loss, inter_s_loss

    def predict_targets(self, target_ids, seq_lens):
        target_ids, embedding_output, attention_mask, predict_mask = self.embed_target(target_ids, seq_lens)
        target_inner = self.forward(embedding_output, attention_mask, end_layer_idx=self.translate_layer_idx)
        output = self.forward(target_inner, attention_mask, start_layer_idx=self.translate_layer_idx)
        output = self.target_out_layer(output).transpose(1, 2)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(output, target_ids.long())  # shape: (N, T)
        to_predict = (1 - predict_mask) * attention_mask  # shape: (N, T)
        loss = torch.mean(masked_reduce_mean(loss, to_predict))

        self.update_shadow_variable(self.shadow_target_inner, target_inner, attention_mask)
        return loss

    def predict_targets_from_feats(
            self,
            feats, feat_lens,
            target_ids, target_lens,
    ):
        feats, feat_embedding, feat_attention_mask, _ = self.embed_feats(feats, feat_lens, mask_lm=False)
        feat_inner = self.forward(feat_embedding, feat_attention_mask, end_layer_idx=self.translate_layer_idx)
        self.update_shadow_variable(self.shadow_feat_inner, feat_inner, feat_attention_mask)

        target_ids, target_embedding, target_attention_mask, _ = self.embed_target(target_ids, target_lens, mask_lm=False)
        target_inner = self.forward(target_embedding, target_attention_mask, end_layer_idx=self.translate_layer_idx)
        self.update_shadow_variable(self.shadow_target_inner, target_inner, target_attention_mask)

        translated_feat_inner = self.translate(feat_inner)
        output = self.forward(translated_feat_inner, feat_attention_mask, start_layer_idx=self.translate_layer_idx)
        output = self.target_out_layer(output)
        return output

    def forward(self, embedding_output, attention_mask, start_layer_idx=0, end_layer_idx=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(embedding_output[:, :, 0])

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        x = embedding_output
        for layer in self.encoder.layer[start_layer_idx:end_layer_idx]:
            x = layer(x, extended_attention_mask)
        return x

    def update_shadow_variable(self, shadow_var, inner, mask):
        mean_inner = torch.mean(
            masked_reduce_mean(inner, mask),
            dim=0,
        )
        shadow_var *= self.update_ratio
        shadow_var += (1 - self.update_ratio) * mean_inner

    def translate(self, feat_inner):
        translation_vec = (self.shadow_target_inner - self.shadow_feat_inner).detach()
        feat_inner = feat_inner + translation_vec
        feat_inner = self.translation_dense(feat_inner)
        return feat_inner

    def embed_target(self, target_ids, seq_lens, mask_lm=True):
        target_ids = get_tensor_from_array(target_ids).long()
        attention_mask = get_attention_mask(seq_lens, target_ids.shape[1])
        predict_mask = None
        if mask_lm:
            input_mask, predict_mask = get_mlm_masks(target_ids, self.mask_prob, self.mask_but_no_prob)

        embedding_output = self.target_embeddings(target_ids)
        embedding_output = self.besides_word_embed(embedding_output)

        if mask_lm:
            input_mask = input_mask.unsqueeze(2)
            embedding_output = self.use_pretrained_mask_embedding(embedding_output, input_mask)

        target_ids = self.pad_front(target_ids, 0)
        embedding_output = self.pad_sep_embedding(embedding_output)
        attention_mask = self.pad_front(attention_mask, 1)
        predict_mask = self.pad_front(predict_mask, 0)
        return target_ids, embedding_output, attention_mask, predict_mask

    def besides_word_embed(self, words_embeddings):
        seq_length = words_embeddings.shape[1]
        batch_size = words_embeddings.shape[0]
        token_type_ids = torch.zeros(batch_size, seq_length)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_type_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_type_ids)

        # words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.pretrained_embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.pretrained_embeddings.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.pretrained_embeddings.LayerNorm(embeddings)
        embeddings = self.pretrained_embeddings.dropout(embeddings)
        return embeddings

    def embed_feats(self, feats, seq_lens, mask_lm=True):
        feats = get_tensor_from_array(feats)
        attention_mask = get_attention_mask(seq_lens, feats.shape[1])
        predict_mask = None
        if mask_lm:
            input_mask, predict_mask = get_mlm_masks(feats, self.mask_prob, self.mask_but_no_prob)

        embedding_output = self.feat_embeddings(feats)

        if mask_lm:
            embedding_output = self.use_pretrained_mask_embedding(embedding_output, input_mask)

        feats = self.pad_front(feats, 0)
        embedding_output = self.pad_sep_embedding(embedding_output, seq_lens)
        attention_mask = self.pad_back(attention_mask, 1)
        predict_mask = self.pad_back(predict_mask, 0)
        embedding_output = self.pad_cls_embedding(embedding_output)
        attention_mask = self.pad_front(attention_mask, 1)
        predict_mask = self.pad_front(predict_mask, 0)
        return feats, embedding_output, attention_mask, predict_mask

    def use_pretrained_mask_embedding(self, inp, input_mask):
        mask_embedding = self.get_pretrained_embeddings(self.mask_token)
        inp = inp * input_mask + (1 - input_mask) * mask_embedding
        return inp

    def pad_cls_embedding(self, inp):
        cls_embedding = self.get_pretrained_embeddings(self.cls_token)
        cls_embeddings = cls_embedding.repeat(inp.shape[0], 1, 1)
        inp = torch.cat([cls_embeddings, inp], dim=1)  # concat [cls] embeddings
        return inp

    def pad_sep_embedding(self, inp, seq_lens):
        sep_embedding = self.get_pretrained_embeddings(self.sep_token)
        sep_embeddings_column = sep_embedding.repeat(inp.shape[0], 1, 1)
        inp = torch.cat([inp, sep_embeddings_column], dim=1)  # concat [sep] embeddings
        inp[torch.range(len(seq_lens)), seq_lens] = sep_embedding
        return inp

    def get_pretrained_embeddings(self, token):
        return self.pretrained_embeddings.word_embeddings(get_tensor_from_array(np.array([[token]])).long())  # shape: (1, E)

    @staticmethod
    def pad_front(tensor, value):
        return torch.cat([
            torch.full_like(tensor[:, :1], value),
            tensor
        ], dim=1)

    @staticmethod
    def pad_back(tensor, value):
        return torch.cat([
            tensor,
            torch.full_like(tensor[:, :1], value)
        ], dim=1)

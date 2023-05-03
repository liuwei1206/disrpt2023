
import math
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import nn
from torch.nn import CrossEntropyLoss
# I modified the name of the pakege from TorchCRF to torchcrf, since my computer has only this version of crf...
from torchcrf import CRF
#from TorchCRF import CRF

from transformers import PreTrainedModel
from transformers.models.roberta import RobertaModel
from transformers.models.bert import BertModel
from transformers.models.electra import ElectraModel
from transformers.models.xlm_roberta import XLMRobertaModel

class BaseRelClassifier(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "electra":
            self.encoder = ElectraModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "xlm-roberta":
            self.encoder = XLMRobertaModel.from_pretrained(args.pretrained_path, config=config)
        self.classifier = nn.Linear(config.hidden_size+args.feature_size, args.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels
        self.do_freeze = args.do_freeze
        self.do_adv = args.do_adv
        self.feature_size = args.feature_size

        if self.do_freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def pretrained_forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        embedding_output = self.encoder.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        input_shape = input_ids.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
        encoder_outputs = self.encoder.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.encoder.pooler(sequence_output)
        return pooled_output, embedding_output

    def adv_attack(self, embedding_output, loss, epsilon=1):
        """
        We choose the direction that makes the loss decreases fastest.

        epsilon = 1 or 5

        refer to: https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/bat_asc.py
        """
        loss_grad = grad(loss, embedding_output, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_embedding_output = embedding_output + epsilon * (loss_grad / (loss_grad_norm.reshape(-1, 1, 1)))
        return perturbed_embedding_output

    def adversarial_forward(self, embedding_output, perturbed_embedding_output, attention_mask, labels, features=None):
        reserve_cls_mask = torch.zeros_like(embedding_output).to(embedding_output.device)
        reserve_cls_mask[:, 0, :] = 1
        perturbed_embedding_output = torch.where(reserve_cls_mask.byte(), embedding_output, perturbed_embedding_output)

        input_shape = perturbed_embedding_output.size()[:2]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, perturbed_embedding_output.device)
        encoder_outputs = self.encoder.encoder(
            perturbed_embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.encoder.pooler(sequence_output)
        if features is not None and self.feature_size > 0:
            pooled_output = torch.cat((pooled_output, features), dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        features=None,
        labels=None,
        flag="Train"
    ):
        pooled_output, embedding_output = self.pretrained_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if features is not None and self.feature_size > 0:
            pooled_output = torch.cat((pooled_output, features), dim=0)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        preds = torch.argmax(logits, dim=-1)
        outputs = (preds,)

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.do_adv:
                perturbed_embedding_output = self.adv_attack(embedding_output, loss)
                adv_loss = self.adversarial_forward(
                    embedding_output,
                    perturbed_embedding_output,
                    attention_mask,
                    labels,
                    features=features,
                )
                loss = loss + adv_loss
            outputs = (loss,) + outputs

        return outputs


class BaseSegClassifier(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "electra":
            self.encoder = ElectraModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "xlm-roberta":
            self.encoder = XLMRobertaModel.from_pretrained(args.pretrained_path, config=config)
        self.classifier = nn.Linear(config.hidden_size+args.feature_size, args.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels
        self.do_freeze = args.do_freeze
        self.do_adv = args.do_adv
        self.feature_size = args.feature_size

        if self.do_freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def pretrained_forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        embedding_output = self.encoder.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        input_shape = input_ids.size()
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
        encoder_outputs = self.encoder.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        return sequence_output, embedding_output

    def adv_attack(self, embedding_output, loss, epsilon=1):
        """
        We choose the direction that makes the loss decreases fastest.

        epsilon = 1 or 5

        refer to: https://github.com/akkarimi/BERT-For-ABSA/blob/master/src/bat_asc.py
        """
        loss_grad = grad(loss, embedding_output, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_embedding_output = embedding_output + epsilon * (loss_grad / (loss_grad_norm.reshape(-1, 1, 1)))
        return perturbed_embedding_output

    def adversarial_forward(self, embedding_output, perturbed_embedding_output, attention_mask, labels, features=None):
        reserve_cls_mask = torch.zeros_like(embedding_output).to(embedding_output.device)
        reserve_cls_mask[:, 0, :] = 1
        perturbed_embedding_output = torch.where(reserve_cls_mask.byte(), embedding_output, perturbed_embedding_output)

        input_shape = perturbed_embedding_output.size()[:2]
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, perturbed_embedding_output.device)
        encoder_outputs = self.encoder.encoder(
            perturbed_embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        if features is not None and self.feature_size > 0:
            sequence_output = torch.cat((sequence_output, features), dim=-1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        features=None,
        labels=None,
        flag="Train"
    ):
        sequence_output, embedding_output = self.pretrained_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if features is not None and self.feature_size > 0:
            sequence_output = torch.cat((sequence_output, features), dim=0)
        pooled_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        preds = torch.argmax(logits, dim=-1)
        outputs = (preds,)

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.do_adv:
                perturbed_embedding_output = self.adv_attack(embedding_output, loss)
                adv_loss = self.adversarial_forward(
                    embedding_output,
                    perturbed_embedding_output,
                    attention_mask,
                    labels,
                    features=features,
                )
                loss = loss + adv_loss
            outputs = (loss,) + outputs
        return outputs


class BiLSTMCRF(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        self.num_labels = args.num_labels
        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(args.pretrained_model, config=config)
        elif self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(args.pretrained_model, config=config)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size, hidden_size=config.hidden_size,
            num_layers=2, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(config.hidden_size*2, self.num_labels)
        self.crf = CRF(self.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.do_freeze = args.do_freeze

        if self.do_freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        flag="Train"
    ):
        if self.do_freeze:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        sequence_outputs = outputs[0]
        sequence_outputs, _ = self.lstm(sequence_outputs)
        feats = self.linear(sequence_outputs)

        if flag.lower() == "trian":
            loss = self.crf.forward(feats, labels, attention_mask).mean()
            loss = -loss
            outputs = (loss, )
        else:
            preds = self.crf.viterbi_decode(feats, attention_mask)
            outputs = (preds, )

        return outputs


class BiLSTMCRFPlus(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        self.num_labels = args.num_labels
        self.extra_feat_dim = args.extra_feat_dim
        self.pos1_type = args.pos1_convert
        self.pos2_type = args.pos2_convert

        self.pos1_vocab_len = args.pos1_vocab_len
        self.pos2_vocab_len = args.pos2_vocab_len

        self.pos1_dim = args.pos1_dim
        self.pos2_dim = args.pos2_dim

        self.pos1_encoder = None
        self.pos2_encoder = None

        self.do_train = args.do_train

        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(args.pretrained_path, config=config)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size, hidden_size=config.hidden_size,
            num_layers=2, bidirectional=True, batch_first=True
        )

        if self.pos1_type == "one-hot":
            self.pos1_encoder = self.pos_onthot_encoder
        elif self.pos1_type == "sequence":
            self.pos1_encoder = self.pos_sequence_encoder

        if self.pos2_type == "one-hot":
            self.pos2_encoder = self.pos_onthot_encoder
        elif self.pos2_type == "sequence":
            self.pos2_encoder = self.pos_sequence_encoder

        self.linear_mid = nn.Linear(config.hidden_size + self.extra_feat_dim, config.hidden_size)

        self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.crf = CRF(self.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.do_freeze = args.do_freeze

        if self.do_freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(
            self,
            input_ids,
            pos1_ids,
            pos2_ids,
            ft_embeds,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            flag="train",
    ):
        if self.do_freeze:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        sequence_outputs = outputs[0]

        pos1_embeds = self.pos1_encoder(pos1_ids, self.pos1_vocab_len, self.pos1_dim)
        pos2_embeds = self.pos2_encoder(pos2_ids, self.pos2_vocab_len, self.pos2_dim)

        # extra_feats = extra_feats.unsqueeze(1)
        sequence_outputs = torch.cat([sequence_outputs, pos1_embeds, pos2_embeds, ft_embeds], dim=-1)

        sequence_outputs = self.linear_mid(sequence_outputs.float())
        sequence_outputs, _ = self.lstm(sequence_outputs)
        feats = self.linear(sequence_outputs)

        if self.do_train:
            attention_mask = torch.BoolTensor(attention_mask)
            labels = labels.long()
            loss = self.crf.forward(feats, labels, attention_mask).mean()
            print("_______________________________________________")
            loss = -loss
            outputs = (loss,)
        else:
            preds = self.crf.viterbi_decode(feats, attention_mask)
            outputs = (preds,)

        return outputs

    def pos_onthot_encoder(self, pos_ids, pos_vocab_len, pos_dim=512):
        res = []
        for i in range(len(pos_ids)):
            for j in range(len(pos_ids[i])):
                tmp_pos = [0] * pos_vocab_len
                tmp_pos[pos_ids[i][j]] = 1
            res.append(np.array(tmp_pos))
        res = np.array(res)
        return res

    def pos_sequence_encoder(self, pos_ids, pos_vocab_len, pos_dim):
        res = []
        pos_encoder = nn.Embedding(pos_vocab_len, pos_dim)
        for i in range(len(pos_ids)):
            tmp = []
            for j in range(len(pos_ids[i])):
                tmp_embeds = pos_encoder(torch.tensor([pos_ids[i][j]]))
                tmp.append(tmp_embeds[0])
            tmp = torch.stack(tmp, dim=0)
            res.append(tmp)
        res = torch.stack(res, dim=0)
        return res

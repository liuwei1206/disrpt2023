
import math
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from TorchCRF import CRF

from transformers import PreTrainedModel
from transformers.models.roberta import RobertaModel
from transformers.models.bert import BertModel

class BaseRelClassifier(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(args.pretrained_path, config=config)
        elif self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(args.pretrained_path, config=config)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels
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
        pooled_outputs = outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs) # [B, N, L] or [B, L]
        preds = torch.argmax(logits, dim=-1)
        outputs = (preds, )

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class BaseSegClassifier(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained("roberta-base", config=config)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels
        self.do_freeze = args.do_freeze

        if self.do_freeze:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        token_level=True,
        flag="Train"
    ):
        if self.do_freeze:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        sequence_outputs = outputs[0]
        sequence_outputs = self.dropout(sequence_outputs)
        logits = self.classifier(sequence_outputs) # [B, N, L] or [B, L]
        preds = torch.argmax(logits, dim=-1)
        outputs = (preds, )

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class BiLSTMCRF(PreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config=config)

        self.encoder_type = args.encoder_type.lower()
        self.num_labels = args.num_labels
        if self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(model_name_or_path["roberta"], config=config)
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
        labels=None,
        flag="Train"
    ):
        if self.do_freeze:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
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











import pandas as pd
import os
import random
import torch
import json
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset
import torch.utils.data as Data
from torch.optim import AdamW


# Dataset
def construct_dataset(file_path, tokenizer, max_len):
    """
        Args:
            file_path: data file path
            tokenizer: tokenizer's instance
            max_len: max length for a pair of tokens
        """
    pair_list = []
    label_list = []
    all_texts = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line_content = json.loads(line)
            all_texts.append(line_content)
    for doc in all_texts:
        doc_token_list = doc["doc_units"]
        doc_label_list = doc["doc_unit_labels"]
        for i in range(len(doc_token_list)):
            pairs = []
            for j in range(len(doc_token_list[i])):
                pairs.append(" ".join(doc_token_list[i][j]))
            pair_list.append(pairs)
            label_list.append(doc_label_list[i])
    input_ids, token_type_ids, attention_mask = [], [], []
    for k in range(len(pair_list)):
        encoded_dict = tokenizer.encode_plus(
            pair_list[k][0],
            pair_list[k][1],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_mask.append(encoded_dict['attention_mask'])

    label2id = {}
    id2label = {}
    id_labels = []
    for id, label in enumerate(np.unique(label_list)):
        label2id[label] = id
        id2label[id] = label
    for l in label_list:
        id_labels.append(label2id[l])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)

    input_ids = torch.LongTensor(input_ids)
    token_type_ids = torch.LongTensor(token_type_ids)
    attention_mask = torch.LongTensor(attention_mask)

    id_labels = torch.LongTensor(id_labels)

    return input_ids, token_type_ids, attention_mask, id_labels

# training
def train_relation(model, input_ids, token_type_ids, attention_mask, labels, epochs, batch_size):

    train_data = Data.TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    train_dataloader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    loss_list = np.array([])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for e in range(epochs):
        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            loss = model(batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
            print(loss.item())
            np.append(loss_list, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# test
def test_relation():
    pass

# assess
def assess_relation():
    pass
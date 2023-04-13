
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class SegDataset(Dataset):
    '''Generate the dataset for task1 Segmentation'''

    def __init__(self, file_name, params):
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_dict = params["label_dict"]

        self._init_dataset(file_name)

    # read the data
    def _init_dataset(self, data_path):
        """
        Args:
            file_name: data path
        """
        default_label = "_"

        token_list = []
        label_list = []
        all_texts = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line_content = json.loads(line)
                all_texts.append(line_content)
        for doc in all_texts:
            doc_token_list = doc["doc_sents"]
            doc_label_list = doc["doc_sent_token_labels"]
            for i in range(len(doc_token_list)):
                for j in range(len(doc_token_list[i])):
                    token_list.append(doc_token_list[i][j])
                    label_list.append(doc_label_list[i][j])

        self.sents, self.labels, self.ids = [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks = [], [], [], [], [], []

        for token, tag in zip(token_list, label_list):
            if token != '.':
                tmp_words.append(token)
                tmp_labels.append(tag)
            else:
                if len(tmp_words) > self.max_seq_length:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words[:self.max_seq_length] + [
                        self.tokenizer.sep_token]
                    temp_label = [default_label] + tmp_labels[:self.max_seq_length] + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_label)
                else:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                    temp_label = [default_label] + tmp_labels + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_label)

                    # convert to ids
                    tmp_tok_ids = self.tokenizer.convert_tokens_to_ids(temp_sent)
                    tmp_label_ids = [self.label_dict[l] for l in temp_label]

                    assert len(tmp_tok_ids) == len(tmp_label_ids), (len(tmp_tok_ids), len(tmp_label_ids))

                    # unify the sequence length
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    label_ids = np.ones(self.max_seq_length, dtype=np.int)

                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(tmp_tok_ids)] = tmp_tok_ids
                    attention_mask[:len(tmp_tok_ids)] = 1
                    label_ids[:len(tmp_label_ids)] = tmp_label_ids

                    # put together
                    tmp_sent_token_ids.append(input_ids)
                    tmp_label_ids_list.append(label_ids)
                    tmp_masks.append(attention_mask)

                tmp_words, tmp_labels = [], []

        self.input_ids = np.array(tmp_sent_token_ids)
        self.attention_mask = np.array(tmp_masks)
        self.label_ids = np.array(tmp_label_ids_list)
        self.total_size = len(tmp_sent_token_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        mask = (self.attention_mask[index] > 0)
        return self.input_ids[index], mask, self.label_ids[index]


class RelDataset(Dataset):
    def __init__(self, file_name, params):
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_dict = params["label_dict"]

        self._init_dataset(file_name)

    def _init_dataset(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            all_texts = f.readlines()

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_label_ids = []
        label_frequency = defaultdict(int)

        for text in all_texts:
            text = text.strip()
            if text:
                sample = json.loads(text)
                doc_units = sample["doc_units"]
                doc_unit_labels = sample["doc_unit_labels"]

                for unit_words, unit_label in zip(doc_units, doc_unit_labels):
                    if unit_label not in self.label_dict:
                        continue
                    unit1 = unit_words[0]
                    unit2 = unit_words[1]
                    arg1 = " ".join(unit1)
                    arg2 = " ".join(unit2)
                    # print(arg1)
                    # print(arg2)
                    res = self.tokenizer(
                        text=arg1,
                        text_pair=arg2,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_tensors="pt"
                    )
                    input_ids = res.input_ids[0]
                    attention_mask = res.attention_mask[0]
                    token_type_ids = res.token_type_ids[0]
                    label_frequency[unit_label] += 1
                    if unit_label in self.label_dict:
                        label_id = self.label_dict[unit_label]
                    else:
                        label_id = 0

                    # put together
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_token_type_ids.append(token_type_ids)
                    all_label_ids.append(label_id)

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.token_type_ids = all_token_type_ids
        self.label_ids = np.array(all_label_ids)
        # print(all_label_ids)
        print(label_frequency)
        self.total_size = len(all_input_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.attention_mask[index],
            self.token_type_ids[index],
            torch.tensor(self.label_ids[index]),
        )





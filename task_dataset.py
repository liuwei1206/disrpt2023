
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class SegDataset(Dataset):
    def __init__(self, file_name, params):
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_dict = params["label_dict"]

        self._init_dataset(file_name)

    def _init_dataset(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            all_texts = f.readlines()

        default_label = "-"
        all_input_ids = []
        all_attention_mask = []
        all_label_ids = []
        all_restore_ids = []

        for text in all_texts:
            text = text.strip()
            if text:
                sample = json.loads(text)
                doc_sents = sample["doc_sents"]
                doc_sent_token_labels = sample["doc_sent_token_labels"]

                for sent_words, word_labels in zip(doc_sents, doc_sent_token_labels):
                    tmp_sent_tokens = [self.tokenizer.cls_token]
                    tmp_sent_token_labels = [default_label] # default label is "-"
                    tmp_restore_pos = []
                    for word, label in zip(sent_words, word_labels):
                        tmp_restore_pos.append(len(tmp_sent_tokens))
                        tmp_tokens = self.tokenizer.tokenize(word)
                        tmp_labels = [default_label for _ in range(len(tmp_tokens)-1)]
                        # tmp_labels.insert(0, label.lower()) # we only preserve the first subtoken's label
                        tmp_labels.insert(0, label)
                        tmp_sent_tokens.extend(tmp_tokens)
                        tmp_sent_token_labels.extend(tmp_labels)
                    if len(tmp_sent_tokens) > self.max_seq_length-1:
                        tmp_sent_tokens = tmp_sent_tokens[:self.max_seq_length-1]
                        tmp_sent_token_labels = tmp_sent_token_labels[:self.max_seq_length-1]
                    tmp_sent_tokens.append(self.tokenizer.sep_token)
                    tmp_sent_token_labels.append(default_label)

                    # convert to ids
                    tmp_sent_token_ids = self.tokenizer.convert_tokens_to_ids(tmp_sent_tokens)
                    tmp_sent_token_label_ids = [self.label_dict[l] for l in tmp_sent_token_labels]
                    assert len(tmp_sent_token_ids) == len(tmp_sent_token_label_ids), (len(tmp_sent_token_ids), len(tmp_sent_token_label_ids))

                    # unify the sequence length
                    input_ids = np.ones(self.max_seq_length, dtype=np.int)
                    attention_mask = np.zeros(self.max_seq_length, dtype=np.int)
                    label_ids = np.ones(self.max_seq_length, dtype=np.int)
                    restore_ids = np.ones(self.max_seq_length, dtype=np.int)

                    input_ids = input_ids * self.tokenizer.pad_token_id
                    input_ids[:len(tmp_sent_token_ids)] = tmp_sent_token_ids
                    attention_mask[:len(tmp_sent_token_ids)] = 1
                    label_ids[:len(tmp_sent_token_label_ids)] = tmp_sent_token_label_ids
                    restore_ids[:len(tmp_restore_pos)] = tmp_restore_pos

                    # put together
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_label_ids.append(label_ids)
                    all_restore_ids.append(restore_ids)


        self.input_ids = np.array(all_input_ids)
        self.attention_mask = np.array(all_attention_mask)
        self.label_ids = np.array(all_label_ids)
        self.restore_ids = np.array(all_restore_ids)

        self.total_size = len(all_input_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.restore_ids[index]),
        )


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

        for text in all_texts:
            text = text.strip()
            if text:
                sample = json.loads(text)
                doc_units = sample["doc_units"]
                doc_unit_labels = sample["doc_unit_labels"]

                for unit_words, unit_label in zip(doc_units, doc_unit_labels):
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





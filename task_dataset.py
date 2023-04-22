
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from utils import encode_words, get_similarity_features

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
        print(data_path)
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
                # print("in-")
            else:
                if len(tmp_words) > self.max_seq_length - 2:
                    print(tmp_words)
                    temp_sent = [self.tokenizer.cls_token] + tmp_words[:self.max_seq_length-2] + [
                        self.tokenizer.sep_token]
                    temp_label = [default_label] + tmp_labels[:self.max_seq_length-2] + [default_label]
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
                input_ids = np.ones(self.max_seq_length, dtype=np.int32)
                attention_mask = np.zeros(self.max_seq_length, dtype=np.int32)
                label_ids = np.ones(self.max_seq_length, dtype=np.int32)
                input_ids = input_ids * self.tokenizer.pad_token_id
                input_ids[:len(tmp_tok_ids)] = tmp_tok_ids
                attention_mask[:len(tmp_tok_ids)] = 1
                label_ids[:len(tmp_label_ids)] = tmp_label_ids

                # put together
                tmp_sent_token_ids.append(input_ids)
                tmp_label_ids_list.append(label_ids)
                tmp_masks.append(attention_mask)

                tmp_words, tmp_labels = [], []
        print("sample size: ", len(tmp_sent_token_ids))
        self.input_ids = np.array(tmp_sent_token_ids)
        self.attention_mask = np.array(tmp_masks)
        self.label_ids = np.array(tmp_label_ids_list)
        self.total_size = len(tmp_sent_token_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        mask = (self.attention_mask[index] > 0)
        return (
            torch.tensor(self.input_ids[index]).long(), 
            torch.tensor(mask).long(), 
            torch.tensor(self.label_ids[index]).long()
        )


class SegDataset2(Dataset):
    '''New version dataset for the task1 and task2 it can fix the UNK token problem'''
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

        self.sents, self.labels, self.ids, self.tok_start_idxs = [], [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks, subword_lengths = [], [], [], [], [], [], []

        for token, tag in zip(token_list, label_list):
            if token != '.' and token != '。' and len(tmp_words) < self.max_seq_length - 2:
                tmp_subtoks = self.tokenizer.tokenize(token)
                subword_lengths.append(len(tmp_subtoks))

                tmp_words += tmp_subtoks
                if tag == "Seg=B-Conn":
                    tmp_labels.append(tag)
                    if "Seg=I-Conn" in self.label_dict:
                        tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                else:
                    tmp_labels += [tag] * len(tmp_subtoks)
            else:
                if len(tmp_words) > self.max_seq_length - 2:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words[:self.max_seq_length - 2] + [
                        self.tokenizer.sep_token]
                    temp_sent_label = [default_label] + tmp_labels[:self.max_seq_length - 2] + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_sent_label)
                else:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                    temp_sent_label = [default_label] + tmp_labels + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_sent_label)

                # convert to ids
                tmp_tok_ids = self.tokenizer.convert_tokens_to_ids(temp_sent)
                tmp_label_ids = [self.label_dict[l] for l in temp_sent_label]
                assert len(tmp_tok_ids) == len(tmp_label_ids), (len(tmp_tok_ids), len(tmp_label_ids))

                # store the location of the first part after word piece
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])

                # unify the sequence length
                input_ids = np.ones(self.max_seq_length, dtype=np.int32)
                attention_mask = np.zeros(self.max_seq_length, dtype=np.int32)
                label_ids = np.ones(self.max_seq_length, dtype=np.int32)
                og_tok_ids = np.zeros(self.max_seq_length, dtype=np.int32)

                input_ids = input_ids * self.tokenizer.pad_token_id
                input_ids[:len(tmp_tok_ids)] = tmp_tok_ids
                attention_mask[:len(tmp_tok_ids)] = 1
                label_ids[:len(tmp_label_ids)] = tmp_label_ids
                og_tok_ids[:len(token_start_idxs)] = token_start_idxs
                # put together
                tmp_sent_token_ids.append(input_ids)
                tmp_label_ids_list.append(label_ids)
                tmp_masks.append(attention_mask)
                self.tok_start_idxs.append(og_tok_ids)
                tmp_words, tmp_labels, subword_lengths = [], [], []

        self.input_ids = np.array(tmp_sent_token_ids)
        self.attention_mask = np.array(tmp_masks)
        self.label_ids = np.array(tmp_label_ids_list)
        self.total_size = len(tmp_sent_token_ids)
        self.tok_start_idxs = np.array(self.tok_start_idxs)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        mask = (self.attention_mask[index] > 0)
        return self.input_ids[index], mask, self.label_ids[index], self.tok_start_idxs[index]


class SegDatasetPlus(Dataset):
    '''Generate the dataset for task1 Segmentation'''

    def __init__(self, file_name, params):
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_dict = params["label_dict"]

        self.pos1_dict = params["pos1_dict"]
        self.pos1_lst = params["pos1_list"]
        self.pos1_convert = params["pos1_convert"]
        self.pos2_dict = params["pos2_dict"]
        self.pos2_lst = params["pos2_list"]
        self.pos2_convert = params["pos2_convert"]
        self.extra_feat_len = 0

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

        pos1_list = []
        pos2_list = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line_content = json.loads(line)
                all_texts.append(line_content)
        for doc in all_texts:
            doc_token_list = doc["doc_sents"]
            doc_label_list = doc["doc_sent_token_labels"]
            doc_pos_list = doc['doc_sent_token_features']

            for i in range(len(doc_token_list)):
                for j in range(len(doc_token_list[i])):
                    token_list.append(doc_token_list[i][j])
                    label_list.append(doc_label_list[i][j])
                    pos1_list.append(doc_pos_list[i][j][1])
                    pos2_list.append(doc_pos_list[i][j][2])

        self.sents, self.labels, self.ids, self.tok_start_idxs, self.tok_pos_list1, self.tok_pos_list2 = [], [], [], [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks, subword_lengths, tmp_tok_pos1, tmp_tok_pos2 = [], [], [], [], [], [], [], [], []

        for i in range(len(token_list)):
            token = token_list[i]
            tag = label_list[i]
            tok_pos1 = pos1_list[i]
            tok_pos2 = pos2_list[i]
            if token != '.' and token != '。' and len(tmp_words) < self.max_seq_length - 2:
                tmp_subtoks = self.tokenizer.tokenize(token)
                subword_lengths.append(len(tmp_subtoks))
                tmp_words += tmp_subtoks
                tmp_tok_pos1 += [tok_pos1] * len(tmp_subtoks)
                tmp_tok_pos2 += [tok_pos2] * len(tmp_subtoks)
                if tag == "Seg=B-Conn":
                    tmp_labels.append(tag)
                    if "Seg=I-Conn" in self.label_dict:
                        tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                else:
                    tmp_labels += [tag] * len(tmp_subtoks)

            else:
                if len(tmp_words) > self.max_seq_length - 2:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words[:self.max_seq_length - 2] + [
                        self.tokenizer.sep_token]
                    temp_sent_label = [default_label] + tmp_labels[:self.max_seq_length - 2] + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_sent_label)
                else:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                    temp_sent_label = [default_label] + tmp_labels + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_sent_label)
                # convert to ids
                if self.pos1_convert == "one-hot":
                    tmp_onehot = [0] * len(self.pos1_lst)
                    for i in range(len(self.pos1_lst)):
                        if self.pos1_lst[i] in tmp_tok_pos1:
                            tmp_onehot[i] = 1
                    self.tok_pos_list1.append(tmp_onehot)
                elif self.pos1_convert == "sequence":
                    tmp_tok_pos1 = [self.pos1_dict[id] for id in tmp_tok_pos1]
                    # fix the length
                    tmp_p1 = np.zeros(self.max_seq_length - 2, dtype=np.int32)
                    tmp_p1[:len(tmp_tok_pos1)] = tmp_tok_pos1
                    self.tok_pos_list1.append(tmp_p1)

                if self.pos2_convert == "one-hot":
                    tmp_onehot = [0] * len(self.pos2_lst)
                    for i in range(len(self.pos2_lst)):
                        if self.pos2_lst[i] in tmp_tok_pos2:
                            tmp_onehot[i] = 1
                    self.tok_pos_list2.append(tmp_onehot)
                elif self.pos2_convert == "sequence":
                    tmp_tok_pos2 = [self.pos2_dict[id] for id in tmp_tok_pos2]
                    # fix the length
                    tmp_p2 = np.zeros(self.max_seq_length - 2, dtype=np.int32)
                    tmp_p2[:len(tmp_tok_pos2)] = tmp_tok_pos2
                    self.tok_pos_list2.append(tmp_p2)

                tmp_tok_ids = self.tokenizer.convert_tokens_to_ids(temp_sent)
                tmp_label_ids = [self.label_dict[l] for l in temp_sent_label]
                assert len(tmp_tok_ids) == len(tmp_label_ids), (len(tmp_tok_ids), len(tmp_label_ids))

                # store the location of the first part after word piece
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])

                # unify the sequence length
                input_ids = np.ones(self.max_seq_length, dtype=np.int32)
                attention_mask = np.zeros(self.max_seq_length, dtype=np.int32)
                label_ids = np.ones(self.max_seq_length, dtype=np.int32)
                og_tok_ids = np.zeros(self.max_seq_length, dtype=np.int32)

                input_ids = input_ids * self.tokenizer.pad_token_id
                input_ids[:len(tmp_tok_ids)] = tmp_tok_ids
                attention_mask[:len(tmp_tok_ids)] = 1
                label_ids[:len(tmp_label_ids)] = tmp_label_ids
                og_tok_ids[:len(token_start_idxs)] = token_start_idxs
                # put together
                tmp_sent_token_ids.append(input_ids)
                tmp_label_ids_list.append(label_ids)
                tmp_masks.append(attention_mask)
                self.tok_start_idxs.append(og_tok_ids)
                tmp_words, tmp_labels, subword_lengths, tmp_tok_pos1, tmp_tok_pos2 = [], [], [], [], []

        self.input_ids = np.array(tmp_sent_token_ids)
        self.attention_mask = np.array(tmp_masks)
        self.label_ids = np.array(tmp_label_ids_list)
        self.total_size = len(tmp_sent_token_ids)
        self.tok_start_idxs = np.array(self.tok_start_idxs)
        self.tok_pos_list1 = np.array(self.tok_pos_list1)
        self.tok_pos_list2 = np.array(self.tok_pos_list2)
        self.extra_feat_len += len(self.tok_pos_list1[0])
        self.extra_feat_len += len(self.tok_pos_list2[0])

    def add_extra_features(self, add_prev_sent=False, add_next_sent=False, add_pos1=False, add_pos2=False,
                           add_fastText=False):
        res = []
        if add_prev_sent:
            pass
        if add_next_sent:
            pass
        if add_pos1:
            res.append(self.tok_pos_list1)
        if add_pos2:
            res.append(self.tok_pos_list2)
        return res

    def get_extra_feat_len(self):
        return self.extra_feat_len

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        mask = (self.attention_mask[index] > 0)
        extra_feats = np.concatenate((self.tok_pos_list1[1], self.tok_pos_list2[1]))
        return self.input_ids[index], mask, self.label_ids[index], self.tok_start_idxs[index], extra_feats


def token_labels_from_file(file_name):
    labels = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_labels"]
                for sent_token_labels in doc_sent_token_labels:
                    for l in sent_token_labels:
                        # labels.add(l.lower())
                        labels.add(l)
    labels = list(labels)
    labels = sorted(labels)
    print(" Total label number: %d\n" % (len(labels)))
    label_dict = {l: idx for idx, l in enumerate(labels)}
    label_id_dict = {idx: l for idx, l in enumerate(labels)}
    return label_dict, label_id_dict, labels


class RelDataset(Dataset):
    def __init__(self, file_name, params):
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_dict = params["label_dict"]
        self.encoder = params["encoder"]

        self._init_dataset(file_name)

    def _init_dataset(self, file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            all_texts = f.readlines()

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_sim_features = []
        all_label_ids = []
        label_frequency = defaultdict(int)
        all_connectives = open("data/dataset/connectives.txt", "r", encoding="utf-8").readlines()
        all_connectives = [conn.strip() for conn in all_connectives]
        conn_reps = encode_words(all_connectives, self.encoder, self.tokenizer, 10)

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
                    sim_features = get_similarity_features(unit1, unit2, conn_reps, self.encoder, self.tokenizer)

                    arg1 = " ".join(unit1)
                    arg2 = " ".join(unit2)
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
                    all_sim_features.append(sim_features)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_token_type_ids.append(token_type_ids)
                    all_label_ids.append(label_id)

        self.sim_features = all_sim_features
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
            self.sim_features[index],
        )





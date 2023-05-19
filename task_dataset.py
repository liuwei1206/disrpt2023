
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from utils import unify_rel_labels
import random


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

        sent_list = []
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
                sent_list.append(doc_token_list[i])
                label_list.append(doc_label_list[i])

        self.sents, self.labels, self.ids = [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks = [], [], [], [], [], []

        for tokens, tags in zip(sent_list, label_list):

            if len(tokens) > self.max_seq_length - 2:
                temp_sent = [self.tokenizer.cls_token] + tokens[:self.max_seq_length - 2] + [self.tokenizer.sep_token]
                temp_label = [default_label] + tags[:self.max_seq_length - 2] + [default_label]
                self.sents.append(temp_sent)
                self.labels.append(temp_label)
            else:
                temp_sent = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                temp_label = [default_label] + tags + [default_label]
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

        self.input_ids = np.array(tmp_sent_token_ids)
        self.attention_mask = np.array(tmp_masks)
        self.label_ids = np.array(tmp_label_ids_list)
        self.total_size = len(tmp_sent_token_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        mask = (self.attention_mask[index] > 0)
        return self.input_ids[index], mask, self.label_ids[index]


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

        sent_list = []
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
                sent_list.append(doc_token_list[i])
                label_list.append(doc_label_list[i])

        self.sents, self.labels, self.ids, self.tok_start_idxs = [], [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks, subword_lengths = [], [], [], [], [], [], []

        for tokens, tags in zip(sent_list, label_list):
            for t in range(len(tokens)):
                tmp_subtoks = self.tokenizer.tokenize(tokens[t])
                subword_lengths.append(len(tmp_subtoks))
                tmp_words += tmp_subtoks
                if tags[t] == "Seg=B-Conn":
                    tmp_labels.append(tags[t])
                    if "Seg=I-Conn" in self.label_dict and "tha" not in data_path:
                        tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                else:
                    tmp_labels += [tags[t]] * len(tmp_subtoks)

            if len(tmp_words) > self.max_seq_length - 2:
                print("too long here------------------------------------------------------------------------")
                print(len(tmp_words))
                temp_sent_1 = [self.tokenizer.cls_token] + tmp_words[:self.max_seq_length - 2] + [
                    self.tokenizer.sep_token]
                temp_sent_label_1 = [default_label] + tmp_labels[:self.max_seq_length - 2] + [default_label]

                temp_sent_2 = [self.tokenizer.cls_token] + tmp_words[self.max_seq_length - 2:] + [
                    self.tokenizer.sep_token]
                temp_sent_label_2 = [default_label] + tmp_labels[self.max_seq_length - 2:] + [default_label]

                self.sents.append(temp_sent_1)
                self.labels.append(temp_sent_label_1)

                self.sents.append(temp_sent_2)
                self.labels.append(temp_sent_label_2)
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

class SegDataset3(Dataset):
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

        sent_list = []
        label_list = []
        all_texts = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line_content = json.loads(line)
                all_texts.append(line_content)
        for doc in all_texts:
            doc_token_list = doc["doc_sents"]
            doc_label_list = doc["doc_sent_token_labels"]

            # serious bug in the rus.rst.rrt data
            # in test tok file line 18988, it's an unknown string, cannot read and operate
            if doc["doc_id"] == "sci.comp_53" and "rus.rst.rrt_test" in data_path:
                doc_token_list[0][0] = "-"
            for i in range(len(doc_token_list)):
                # bugs in tur.pdtb.tdb
                if "tur.pdtb.tdb" in data_path:
                    for t in range(len(doc_token_list[i])):
                        if doc_token_list[i][t] == "":
                             doc_token_list[i][t] = "-"
                if "spa.rst.rststb" in data_path:
                    for t in range(len(doc_token_list[i])):
                        if doc_token_list[i][t] == "\x91":
                             doc_token_list[i][t] = "_"
                sent_list.append(doc_token_list[i])
                label_list.append(doc_label_list[i])


        self.sents, self.labels, self.ids, self.tok_start_idxs = [], [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks, subword_lengths = [], [], [], [], [], [], []

        for tokens, tags in zip(sent_list, label_list):
            og_index = np.zeros(self.max_seq_length, dtype=np.int32)
            subword_index = 1
            for t in range(len(tokens)):
                truncated_index = 0

                tmp_subtoks = self.tokenizer.tokenize(tokens[t])
                subword_lengths.append(len(tmp_subtoks))
                #if truncated_index + len(tmp_subtoks) < self.max_seq_length - 2 and subword_index + len(tmp_subtoks) < self.max_seq_length - 2:
                #if subword_index + len(tmp_subtoks) < self.max_seq_length - 2:
                if (len(tmp_words) + len(tmp_subtoks)) < self.max_seq_length - 3:
                    truncated_index = truncated_index + len(tmp_subtoks)
                    og_index[subword_index] = 1
                    subword_index += len(tmp_subtoks)

                    tmp_words += tmp_subtoks
                    if tags[t] == "Seg=B-Conn":
                        tmp_labels.append(tags[t])
                        if "Seg=I-Conn" in self.label_dict and "tha" not in data_path:
                            tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                        else:
                            tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += [tags[t]] * len(tmp_subtoks)
                else:
                    temp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                    temp_sent_label = [default_label] + tmp_labels + [default_label]
                    self.sents.append(temp_sent)
                    self.labels.append(temp_sent_label)
                    self.tok_start_idxs.append(og_index)
                    og_index = np.zeros(self.max_seq_length, dtype=np.int32)
                    subword_index = 1
                    og_index[subword_index] = 1
                    tmp_words, tmp_labels = [], []
                    tmp_words += tmp_subtoks

                    if tags[t] == "Seg=B-Conn":
                        tmp_labels.append(tags[t])
                        if "Seg=I-Conn" in self.label_dict and "tha" not in data_path:
                            tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                        else:
                            tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += [tags[t]] * len(tmp_subtoks)
            if len(tmp_words) > 0:
                temp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                temp_sent_label = [default_label] + tmp_labels + [default_label]
                self.sents.append(temp_sent)
                self.labels.append(temp_sent_label)
                self.tok_start_idxs.append(og_index)

                tmp_words, tmp_labels = [], []

        for sent, label in zip(self.sents, self.labels):
            # convert to ids
            tmp_tok_ids = self.tokenizer.convert_tokens_to_ids(sent)
            tmp_label_ids = [self.label_dict[l] for l in label]
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
            tmp_words, tmp_labels, subword_lengths = [], [], []
            too_long_flag = False
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

class SegDataset4Bag(Dataset):
    '''New version dataset for the task1 and task2 it can fix the UNK token problem'''
    '''Generate the dataset for task1 Segmentation'''

    def __init__(self, file_name, params):
        self.max_seq_length = params["max_seq_length"]
        self.tokenizer = params["tokenizer"]
        self.label_dict = params["label_dict"]
        self.ratio = params["ratio"]
        self._init_dataset(file_name)

    # read the data
    def _init_dataset(self, data_path):
        """
        Args:
            file_name: data path
        """
        truncated = False
        default_label = "_"

        sent_list = []
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
                sent_list.append(doc_token_list[i])
                label_list.append(doc_label_list[i])

        num_samples = int(len(sent_list) * self.ratio)
        sampled_indices = random.sample(range(len(sent_list)), num_samples)
        sampled_sent_list = [sent_list[i] for i in sampled_indices]
        sampled_label_list = [label_list[i] for i in sampled_indices]

        self.sents, self.labels, self.ids, self.tok_start_idxs = [], [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks, subword_lengths = [], [], [], [], [], [], []

        for tokens, tags in zip(sampled_sent_list, sampled_label_list):
            for t in range(len(tokens)):
                truncated_index = 0
                tmp_subtoks = self.tokenizer.tokenize(tokens[t])
                subword_lengths.append(len(tmp_subtoks))
                if truncated_index + len(tmp_subtoks) < self.max_seq_length - 2:
                    truncated_index = truncated_index + len(tmp_subtoks)
                tmp_words += tmp_subtoks
                if tags[t] == "Seg=B-Conn":
                    tmp_labels.append(tags[t])
                    if "Seg=I-Conn" in self.label_dict and "tha" not in data_path:
                        tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                else:
                    tmp_labels += [tags[t]] * len(tmp_subtoks)

            temp_sent_list = []
            temp_label_list = []
            if len(tmp_words) > self.max_seq_length - 2:
                print("too long here------------------------------------------------------------------------")
                print(len(tmp_words))
                tmp_sent_1 = [self.tokenizer.cls_token] + tmp_words[:truncated_index] + [
                    self.tokenizer.sep_token]
                tmp_sent_label_1 = [default_label] + tmp_labels[:truncated_index] + [default_label]

                tmp_sent_2 = [self.tokenizer.cls_token] + tmp_words[truncated_index:] + [
                    self.tokenizer.sep_token]
                tmp_sent_label_2 = [default_label] + tmp_labels[truncated_index:] + [default_label]

                self.sents.append(tmp_sent_1)
                self.labels.append(tmp_sent_label_1)

                self.sents.append(tmp_sent_2)
                self.labels.append(tmp_sent_label_2)

                temp_sent_list.append(tmp_sent_1)
                temp_sent_list.append(tmp_sent_2)

                temp_label_list.append(tmp_sent_label_1)
                temp_label_list.append(tmp_sent_label_2)
            else:
                tmp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                tmp_sent_label = [default_label] + tmp_labels + [default_label]
                self.sents.append(tmp_sent)
                self.labels.append(tmp_sent_label)
                temp_sent_list.append(tmp_sent)
                temp_label_list.append(tmp_sent_label)

            # convert to ids
            for temp_sent, temp_sent_label in zip(temp_sent_list, temp_label_list):
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
        # donwload manually would be faster
        # fasttext.util.download_model(params["fasttext_language"], if_exists='ignore')  # English
        # self.ft = fasttext.load_model(params["fasttext_model"])
        # self.ft_dict = np.load('/content/drive/MyDrive/shared_task/eng.rst.gum.npy', allow_pickle=True).item()
        # self.ft_dict = np.load(params["ft_dict"], allow_pickle=True).item()
        self.ft_dict = params["ft_dict"]
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
        default_ft_embeds = np.zeros((1, 300))

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
                tmp_read_toks = []
                tmp_read_tags = []
                tmp_read_pos1 = []
                tmp_read_pos2 = []
                for j in range(len(doc_token_list[i])):
                    tmp_read_toks.append(doc_token_list[i][j])
                    tmp_read_tags.append(doc_label_list[i][j])
                    tmp_read_pos1.append(doc_pos_list[i][j][1])
                    tmp_read_pos2.append(doc_pos_list[i][j][2])
                token_list.append(tmp_read_toks)
                label_list.append(tmp_read_tags)
                pos1_list.append(tmp_read_pos1)
                pos2_list.append(tmp_read_pos2)

        self.sents, self.labels, self.ids, self.tok_start_idxs, self.tok_pos_list1, self.tok_pos_list2, self.fasttext_embeds = [], [], [], [], [], [], []
        tmp_words, tmp_labels, tmp_sent_token_ids, tmp_label_ids, tmp_label_ids_list, tmp_masks, subword_lengths, tmp_tok_pos1, tmp_tok_pos2, tmp_fast_embeds, tmp_fast_embeds_list = [], [], [], [], [], [], [], [], [], [], []

        for i in range(len(token_list)):
            tokens = token_list[i]
            tags = label_list[i]
            # print(pos1_list)
            tok_pos1_list = pos1_list[i]
            tok_pos2_list = pos2_list[i]
            for t in range(len(tokens)):
                tmp_fast_embeds = self.ft_dict[tokens[t]]

                tmp_subtoks = self.tokenizer.tokenize(tokens[t])
                subword_lengths.append(len(tmp_subtoks))
                tmp_words += tmp_subtoks
                tmp_fast_embeds_list += [tmp_fast_embeds] * len(tmp_subtoks)

                tok_pos1 = self.pos1_dict[tok_pos1_list[t]]
                tok_pos2 = self.pos2_dict[tok_pos2_list[t]]

                tmp_tok_pos1 += [tok_pos1] * len(tmp_subtoks)
                tmp_tok_pos2 += [tok_pos2] * len(tmp_subtoks)

                if tags[t] == "Seg=B-Conn":
                    tmp_labels.append(tags[t])
                    if "Seg=I-Conn" in self.label_dict:
                        tmp_labels += ["Seg=I-Conn"] * (len(tmp_subtoks) - 1)
                    else:
                        tmp_labels += ["Seg=B-Conn"] * (len(tmp_subtoks) - 1)
                else:
                    tmp_labels += [tags[t]] * len(tmp_subtoks)

            if len(tmp_words) > self.max_seq_length - 2:
                temp_sent = [self.tokenizer.cls_token] + tmp_words[:self.max_seq_length - 2] + [
                    self.tokenizer.sep_token]
                temp_sent_label = [default_label] + tmp_labels[:self.max_seq_length - 2] + [default_label]
                tmp_tok_pos1 = [0] + tmp_tok_pos1[:self.max_seq_length - 2] + [0]
                tmp_tok_pos2 = [0] + tmp_tok_pos2[:self.max_seq_length - 2] + [0]
                tmp_fast_embeds_list = [default_ft_embeds] + tmp_fast_embeds_list[:self.max_seq_length - 2] + [
                    default_ft_embeds]

                self.sents.append(temp_sent)
                self.labels.append(temp_sent_label)
            else:
                temp_sent = [self.tokenizer.cls_token] + tmp_words + [self.tokenizer.sep_token]
                temp_sent_label = [default_label] + tmp_labels + [default_label]
                tmp_tok_pos1 = [0] + tmp_tok_pos1 + [0]
                tmp_tok_pos2 = [0] + tmp_tok_pos2 + [0]
                tmp_fast_embeds_list = [default_ft_embeds] + tmp_fast_embeds_list + [default_ft_embeds]

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
            pos1_ids = np.zeros(self.max_seq_length, dtype=np.int32)
            pos2_ids = np.zeros(self.max_seq_length, dtype=np.int32)

            tmp_ft_list = [np.zeros((1, 300))] * self.max_seq_length

            input_ids = input_ids * self.tokenizer.pad_token_id
            input_ids[:len(tmp_tok_ids)] = tmp_tok_ids
            attention_mask[:len(tmp_tok_ids)] = 1
            label_ids[:len(tmp_label_ids)] = tmp_label_ids

            og_tok_ids[:len(token_start_idxs)] = token_start_idxs
            pos1_ids[:len(tmp_tok_pos1)] = tmp_tok_pos1
            pos2_ids[:len(tmp_tok_pos2)] = tmp_tok_pos2

            for i in range(len(tmp_ft_list)):
                if i < len(tmp_fast_embeds_list):
                    tmp_ft_list[i] = tmp_fast_embeds_list[i]

            #tmp_ft_list = np.array(tmp_ft_list)
            # put together
            tmp_sent_token_ids.append(input_ids)
            tmp_label_ids_list.append(label_ids)
            tmp_masks.append(attention_mask)
            self.tok_start_idxs.append(og_tok_ids)
            self.tok_pos_list1.append(pos1_ids)
            self.tok_pos_list2.append(pos2_ids)

            self.fasttext_embeds.append(tmp_ft_list)

            tmp_words, tmp_labels, subword_lengths, tmp_tok_pos1, tmp_tok_pos2, tmp_fast_embeds_list = [], [], [], [], [], []

        self.input_ids = np.array(tmp_sent_token_ids)
        self.attention_mask = np.array(tmp_masks)
        self.label_ids = np.array(tmp_label_ids_list)
        self.total_size = len(tmp_sent_token_ids)
        self.tok_start_idxs = np.array(self.tok_start_idxs)
        self.tok_pos_list1 = np.array(self.tok_pos_list1)
        self.tok_pos_list2 = np.array(self.tok_pos_list2)
        # print(len(self.fasttext_embeds))
        # print(len(self.fasttext_embeds[0]))
        # self.fasttext_embeds = np.array(self.fasttext_embeds)

        # print(self.self.fasttext_embeds.shape)

        self.extra_feat_len += len(self.tok_pos_list1[0]) + len(self.tok_pos_list2[0]) + 300

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
        mask = torch.from_numpy(mask)
        ft_embeds = np.vstack(self.fasttext_embeds[index]).astype(float)
        ft_embeds = torch.from_numpy(ft_embeds)
        # extra_feats = np.concatenate((self.tok_pos_list1[index], self.tok_pos_list2[index]))
        # return torch.from_numpy(self.input_ids[index]), torch.from_numpy(mask), torch.from_numpy(self.label_ids[index]), torch.from_numpy(self.tok_start_idxs[index]), torch.from_numpy(self.tok_pos_list1[index]), torch.from_numpy(self.tok_pos_list2[index]), torch.from_numpy(self.fasttext_embeds[index])
        return self.input_ids[index], mask, self.label_ids[index], self.tok_start_idxs[index], self.tok_pos_list1[
            index], self.tok_pos_list2[index], ft_embeds


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
        all_label_index_ids = []
        label_frequency = defaultdict(int)

        for text in all_texts:
            text = text.strip()
            if text:
                sample = json.loads(text)
                dname = sample["dname"]
                doc_units = sample["doc_units"]
                doc_unit_labels = sample["doc_unit_labels"]

                for unit_words, label_info in zip(doc_units, doc_unit_labels):
                    # unit_label = unit_label.lower()
                    unit_label = label_info[0]
                    label_index_id = int(label_info[1])
                    unit_label = unify_rel_labels(unit_label, dname)
                    unit1 = unit_words[0]
                    unit2 = unit_words[1]

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
                    if "token_type_ids" in res:
                        token_type_ids = res.token_type_ids[0]
                    else:
                        token_type_ids = torch.zeros_like(attention_mask)
                    label_frequency[unit_label] += 1
                    if unit_label in self.label_dict:
                        label_id = self.label_dict[unit_label]
                    else:
                        label_id = -100

                    # put together
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_token_type_ids.append(token_type_ids)
                    all_label_ids.append(label_id)
                    all_label_index_ids.append(label_index_id)
        """
        if len(all_input_ids) > 2000:
            all_input_ids = all_input_ids[:1000]
            all_attention_mask = all_attention_mask[:1000]
            all_token_type_ids = all_token_type_ids[:1000]
            all_label_ids = all_label_ids[:1000]
        """

        self.input_ids = all_input_ids
        self.attention_mask = all_attention_mask
        self.token_type_ids = all_token_type_ids
        self.label_ids = np.array(all_label_ids)
        self.label_index_ids = np.array(all_label_index_ids)
        # print(label_frequency)
        self.total_size = len(all_input_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.attention_mask[index],
            self.token_type_ids[index],
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.label_index_ids[index]),
        )



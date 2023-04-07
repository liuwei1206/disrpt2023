import pandas as pd
from glob import glob
import os
import random
import torch
import json
import numpy as np
import shutil
import os
import torch
import torch.nn as nn
# from torchcrf import CRF
from TorchCRF import CRF
from torch.utils.data import Dataset
from transformers import RobertaModel
from transformers import RobertaTokenizer


# minus [CLS] and [SEP]
MAX_LEN = 512 - 2

# Dataset
class SegDataset(Dataset):
    '''Generate the dataset for task1 Segmentation'''

    def __init__(self, data_path, tokenizer):
        """
            Args:
                data_path: data file path
                tokenizer: tokenizer's instance
        """
        self.token_list, self.label_list = self.read_segmentation(data_path)
        self.id2lable_dic, self.label2id_dic, self.original_label = self.transfer_label()
        self.sents = []
        self.labels = []
        self.tokenizer = tokenizer
        words, tags = [], []

        for token, tag in zip(self.token_list, self.label_list):
            if token != '.':
                words.append(token)
                tags.append(tag)
            else:
                if len(words) > MAX_LEN:
                    self.sents.append(['[CLS]'] + words[:MAX_LEN] + ['[SEP]'])
                    self.labels.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
                else:
                    self.sents.append(['[CLS]'] + words + ['[SEP]'])
                    self.labels.append(['[CLS]'] + tags + ['[SEP]'])
                words, tags = [], []
        print("dataset size: ", len(self.sents))

    # read the data
    def read_segmentation(self, data_path):
        """
        Args:
            data_path: path of the data file
        """
        token_list = []
        label_list = []
        all_texts = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                line_content = json.loads(line)
                all_texts.append(line_content)
        all_texts = all_texts[:2]
        for doc in all_texts:
            doc_token_list = doc["doc_sents"]
            doc_label_list = doc["doc_sent_token_labels"]
            for i in range(len(doc_token_list)):
                for j in range(len(doc_token_list[i])):
                    token_list.append(doc_token_list[i][j])
                    label_list.append(doc_label_list[i][j])
        return token_list, label_list

    def transfer_label(self):
        original_label = np.unique(self.label_list)
        label_types = np.append(np.unique(self.label_list), ['<PAD>', '[CLS]', '[SEP]'])
        id2lable_dic = {id: label for id, label in enumerate(label_types)}
        label2id_dic = {label: id for id, label in enumerate(label_types)}
        return id2lable_dic, label2id_dic, original_label

    def __getitem__(self, id):
        words, tags = self.sents[id], self.labels[id]
        token_ids = self.tokenizer.convert_tokens_to_ids(words)
        label_ids = [self.label2id_dic[tag] for tag in tags]
        sequence_len = len(label_ids)
        return token_ids, label_ids, sequence_len

    def __len__(self):
        return len(self.sents)

# for collate_fn
def padding_batch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor([i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask

# Model
class Roberta_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Roberta_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim * 2, self.tagset_size)
        self.crf = CRF(self.tagset_size) # , batch_first=True)

    def _get_features(self, sentence):
        with torch.no_grad():
            embeds = self.roberta(sentence)['last_hidden_state']
        enc, _ = self.lstm(embeds)
        feats = self.linear(enc)
        return feats

    def forward(self, input, mask):
        out = self._get_features(input)

        # return self.crf.decode(out, mask)
        return self.crf.viterbi_decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_features(input)
        return -self.crf.forward(y_pred, target, mask).mean()



# training
def train_segmentation(file_path, model_choice, tokenizer, epochs):

    dataset = SegDataset(file_path, tokenizer)
    label2id = dataset.label2id_dic
    original_label = dataset.original_label
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=padding_batch,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_choice(label2id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epochs):
        for b, batch in enumerate(loader):
            batch = tuple(t.to(device) for t in batch)
            #y_pred = model(batch[0], batch[2])
            loss = model.loss_fn(batch[0], batch[1], batch[2])

            #np.append(loss_list, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 10 == 0:
                print('>> epoch:', e, 'loss:', loss.item())
    return model, label2id, original_label

def test_segmentation(file_path, tokenizer, model):
    dataset = SegDataset(file_path, tokenizer)
    label2id = dataset.label2id_dic
    original_label = dataset.original_label
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=padding_batch,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        y_true_list_train = []
        y_pred_list_train = []
        for b, _batch in enumerate(loader):
            batch = tuple(t.to(device) for t in _batch)
            y_pred = model(batch[0], batch[2])
            # loss = model.loss_fn(input, target, mask)
            for lst in y_pred:
                y_pred_list_train += lst
            for y, m in zip(batch[1], batch[2]):
                y_true_list_train += y[m == True].tolist()

    y_true_tensor_train = torch.tensor(y_true_list_train)
    y_pred_tensor_train = torch.tensor(y_pred_list_train)

    return y_true_tensor_train, y_pred_tensor_train, label2id, original_label

def assess_segmentation(true_label, predict_label, label2id, original_label):
    correct = 0
    precision_list = []
    recall_list = []
    f1_list = []
    id_list = [label2id[i] for i in original_label]
    total = 0
    for label in original_label:
        id = label2id[label]
        TP, FP, FN = 0, 0, 0
        for i in range(len(true_label)):
            if true_label[i] == id and predict_label[i] == id:
                TP += 1
            elif true_label[i] == id and predict_label[i] != id:
                FN += 1
            elif true_label[i] != id and predict_label[i] == id:
                FP += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * recall * precision / (recall + precision)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(F1)
    for j in range(len(true_label)):
        if true_label[j] in id_list:
            total += 1
            if true_label[j] == predict_label[j]:
                correct += 1
    accuracy = correct / total
    print('>> total:', total)
    print('accuracy:', accuracy)
    for k in range(len(original_label)):
        print("precision for " + original_label[k] + " :" + str(precision_list[k]))
        print("recall for " + original_label[k] + " :" + str(recall_list[k]))
        print("f1 score for " + original_label[k] + " :" + str(f1_list[k]))
    print("overall precision:", str(np.mean(precision_list)))
    print("overall recall", str(np.mean(recall_list)))
    print("overall f1:", str(np.mean(f1_list)))
    return precision_list, recall_list, f1_list


if __name__ == "__main__":
    train_file_path = "data/dataset/eng.rst.gum/eng.rst.gum_train.json"
    dev_file_path = "data/dataset/eng.rst.gum/eng.rst.gum_dev.json"
    roberta_model = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
    model_choice = Roberta_BiLSTM_CRF
    epochs = 10

    trained_model, label2id, original_label = train_segmentation(train_file_path, model_choice, tokenizer, epochs)
    
    y_true_tensor_train, y_pred_tensor_train, label2id, original_label = test_segmentation(dev_file_path, tokenizer, trained_model)
    precision_list, recall_list, f1_list = assess_segmentation(y_true_tensor_train, y_pred_tensor_train, label2id, original_label)

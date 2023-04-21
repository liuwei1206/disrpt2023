import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

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
    print(" Total label number: %d\n"%(len(labels)))
    label_dict = {l: idx for idx, l in enumerate(labels)}
    # label_id_dict = {idx: l for idx, l in enumerate(labels)}
    return label_dict, labels

def token_pos_from_file(file_name):
    tok_pos_1 = set()
    tok_pos_2 = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_features"]
                for sent_token_labels in doc_sent_token_labels:
                    for feat in sent_token_labels:
                        tok_pos_1.add(feat[1])
                        tok_pos_2.add(feat[2])
    tok_pos_1 = list(tok_pos_1)
    tok_pos_2 = list(tok_pos_2)
    tok_pos_1 = sorted(tok_pos_1)
    tok_pos_2 = sorted(tok_pos_2)
    tok_pos_1_dict = {t: idx + 1 for idx, t in enumerate(tok_pos_1)}
    tok_pos_2_dict = {t: idx + 1 for idx, t in enumerate(tok_pos_2)}

    return tok_pos_1, tok_pos_2, tok_pos_1_dict, tok_pos_2_dict

def rel_labels_from_file(file_name):
    label_frequency = defaultdict(int)
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_unit_labels = sample["doc_unit_labels"]
                for l in doc_unit_labels:
                    # labels.add(l.lower())
                    label_frequency[l] += 1
    labels = []
    for key in label_frequency:
        if label_frequency[key] >= 0:
            labels.append(key)
    # labels = list(labels)
    labels = sorted(labels)
    label_dict = {l: idx for idx, l in enumerate(labels)}
    print(label_dict)
    # print(labels)
    print(" Total label number: %d\n"%(len(labels)))

    return label_dict, labels

def seg_preds_to_file(all_pred_ids, all_label_ids, all_attention_mask, tokenizer, label_id_dict, gold_file):
    """
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_pred_ids: predicted tokens' id list 
        all_label_ids: predicted labels' id list 
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    all_doc_data = []
    new_doc_data = []
    with open(gold_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc = []
        for line in lines:
            all_doc_data.append(line)
    og_tokens = []
    pred_labels = []

    for i in range(len(all_attention_mask)):
        tmp_toks = tokenizer.convert_ids_to_tokens(all_pred_ids[i])
        for j in range(len(all_attention_mask[i])):
            if all_attention_mask[i][j]:
                # the mapping problem happens here! I fixed it.
                if tmp_toks[j] != "[CLS]" and tmp_toks[j] != "<s>":
                    if tmp_toks[j] == "[SEP]" or tmp_toks[j] == "</s>":
                        og_tokens.append(".")
                        pred_labels.append("_")
                    else:
                        og_tokens.append(tmp_toks[j])
                        pred_labels.append(label_id_dict[int(all_label_ids[i][j])])
    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            if "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                items[-1] = pred_labels[pointer]
                # here, I force items[-2] to be the original token, so you can see from the output file
                # that every token is mapped well. If you check that everything is ok, it can be deleted.
                items[-2] = og_tokens[pointer]
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')

    pred_file = gold_file.replace(".tok", "_pred.tok")
    with open(pred_file,"w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)

    return pred_file

def seg_preds_to_file_new(all_input_ids, all_label_ids, all_attention_mask, all_tok_idxs, tokenizer, label_id_dict, gold_tok_file):
    """
    new version of writing a result tok file
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_input_ids: predicted tokens' id list 
        all_label_ids: predicted labels' id list 
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    all_doc_data = []
    new_doc_data = []
    with open(gold_tok_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc = []
        for line in lines:
            all_doc_data.append(line)
    og_tokens = []
    pred_labels = []

    for i in range(len(all_attention_mask)):
        tmp_idx = [idx for idx in all_tok_idxs[i] if idx > 0]
        point_loc = np.argwhere(all_input_ids[i] == 102)
        if len(point_loc):
            tmp_idx.append(point_loc[0][0])
        og_toks_ids = all_input_ids[i][tmp_idx]
        tmp_toks = tokenizer.convert_ids_to_tokens(og_toks_ids)

        for j in range(len(tmp_toks)):
            if tmp_toks[j] == "[SEP]":
                og_tokens.append(".")
                pred_labels.append("_")
            else:
                og_tokens.append(tmp_toks[j])
                pred_labels.append(label_id_dict[int(all_label_ids[i][j])])

    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            if "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                items[-1] = pred_labels[pointer]
                items[-2] = og_tokens[pointer]
                print(items)
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')

    pred_file = gold_tok_file.replace(".tok", "_pred.tok")
    with open(pred_file,"w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)



def rel_preds_to_file(pred_ids, label_list, gold_file):
    pred_labels = [label_list[id] for id in pred_ids]
    valid_lines = []
    with open(gold_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        title_line = lines[0]
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                valid_lines.append(line)

    assert len(pred_labels) == len(valid_lines), (len(pred_labels), len(valid_lines))

    pred_contents = []
    for pred, line in zip(pred_labels, valid_lines):
        items = line.split("\t")
        new_items = items[:-1]
        new_items.append(pred)
        pred_contents.append("\t".join(new_items))

    pred_file = gold_file.replace(".rels", "_pred.rels")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("%s\n"%(title_line.strip()))
        for text in pred_contents:
            f.write("%s\n"%(text))

    return pred_file

def fix_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model

def encode_words(word_list, encoder, tokenizer, max_length=6):
    res = tokenizer(
        text=word_list,
        padding="max_length",
        truncation=True,
        max_length=max_length, # I guess 6 is enough
        return_tensors="pt"
    )
    input_ids = res.input_ids
    attention_mask = res.attention_mask
    token_type_ids = res.token_type_ids
    inputs = {
        "input_ids": input_ids.to(encoder.device),
        "attention_mask": attention_mask.to(encoder.device),
        "token_type_ids": token_type_ids.to(encoder.device),
    }
    with torch.no_grad():
        outputs = encoder(**inputs)
    word_reps = outputs.pooler_output
    return word_reps

def get_similarity_features(word_list1, word_list2, conn_reps, encoder, tokenizer):
    """
    refer to paper: Discourse Relation Sense Classification Using Cross-argument Semantic Similarity Based on Word Embeddings
    Args:
        word_list1:
        word_list2:
        conn_reps:
        encoder:
        tokenizer:
    Returns:
        features:
    """
    word_num1 = len(word_list1)
    word_num2 = len(word_list2)
    word_list = word_list1 + word_list2
    word_reps = encode_words(word_list, encoder, tokenizer)
    word_reps_1 = word_reps[:word_num1, :] # [N1, D]
    word_reps_2 = word_reps[word_num1:, :] # [N2, D]
    centroid_rep = torch.mean(word_reps, dim=0) # [D]
    centroid_rep_1 = torch.mean(word_reps_1, dim=0) # [D]
    centroid_rep_2 = torch.mean(word_reps_2, dim=0) # [D]

    # 1. calculate similarities
    # 1.1 arg1 to arg2 similarities
    arg1_arg2_score = 1 - F.cosine_similarity(centroid_rep_1, centroid_rep_2, dim=0)

    # 1.2 maximized similarity
    word1_to_arg2_scores = 1 - F.cosine_similarity(centroid_rep_1, word_reps_2, dim=1) # [N1]
    word2_to_arg1_scores = 1 - F.cosine_similarity(centroid_rep_2, word_reps_1, dim=1) # [N2]
    avg_top1_1to2_score = torch.max(word1_to_arg2_scores)
    avg_top1_2to1_score = torch.max(word2_to_arg1_scores)
    if len(word1_to_arg2_scores) > 1:
        top2_1to2_scores = torch.topk(word1_to_arg2_scores, k=2)[0]
        avg_top2_1to2_score = torch.mean(top2_1to2_scores)
    else:
        avg_top2_1to2_score = 0.0
    if len(word1_to_arg2_scores) > 2:
        top3_1to2_scores = torch.topk(word1_to_arg2_scores, k=3)[0]
        avg_top3_1to2_score = torch.mean(top3_1to2_scores)
    else:
        avg_top3_1to2_score = 0.0
    if len(word1_to_arg2_scores) > 4:
        top5_1to2_scores = torch.topk(word1_to_arg2_scores, k=4)[0]
        avg_top5_1to2_score = torch.mean(top5_1to2_scores)
    else:
        avg_top5_1to2_score = 0.0
    if len(word2_to_arg1_scores) > 1:
        top2_2to1_scores = torch.topk(word2_to_arg1_scores, k=2)[0]
        avg_top2_2to1_score = torch.mean(top2_2to1_scores)
    else:
        avg_top2_2to1_score = 0.0
    if len(word2_to_arg1_scores) > 2:
        top3_2to1_scores = torch.topk(word2_to_arg1_scores, k=3)[0]
        avg_top3_2to1_score = torch.mean(top3_2to1_scores)
    else:
        avg_top3_2to1_score = 0.0
    if len(word2_to_arg1_scores) > 4:
        top5_2to1_scores = torch.topk(word2_to_arg1_scores, k=4)[0]
        avg_top5_2to1_score = torch.mean(top5_2to1_scores)
    else:
        avg_top5_2to1_score = 0.0

    # 1.3 aligned similarity
    fenzi = torch.matmul(word_reps_1, word_reps_2.transpose(1, 0))
    fenmu = torch.norm(word_reps_1, p=2, dim=1).unsqueeze(1) * torch.norm(word_reps_2, p=2, dim=1)
    word1_word2_scores = 1 - fenzi / fenmu # [N1, N2]
    avg_word1_word2_score = torch.mean(torch.max(word1_word2_scores, dim=-1)[0])

    # 1.4 conn similarity
    centroid_to_conn_scores = 1 - F.cosine_similarity(centroid_rep, conn_reps, dim=1) # [N1]
    centroid_to_conn_scores = centroid_to_conn_scores.detach().cpu()

    ## 2. merge features
    features = []
    features.append(arg1_arg2_score)
    features.append(avg_top1_1to2_score)
    features.append(avg_top1_2to1_score)
    features.append(avg_top2_1to2_score)
    features.append(avg_top2_2to1_score)
    features.append(avg_top3_1to2_score)
    features.append(avg_top3_2to1_score)
    features.append(avg_top5_1to2_score)
    features.append(avg_top5_2to1_score)
    features.append(avg_word1_word2_score)
    features = torch.tensor(features)
    features = torch.cat((features, centroid_to_conn_scores), dim=-1)

    return features



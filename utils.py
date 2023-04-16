import os
import json
from collections import defaultdict

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
    # print(all_label_ids)
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
                if tmp_toks[j] != "[CLS]":
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
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')

    pred_file = gold_file.replace(".tok", "_pred.tok")
    print(pred_file)
    with open(pred_file,"w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)

    return pred_file

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



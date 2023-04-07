import os
import json

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
    print(labels)
    label_dict = {l: idx for idx, l in enumerate(labels)}

    return label_dict, labels

def rel_labels_from_file(file_name):
    labels = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_unit_labels = sample["doc_unit_labels"]
                for l in doc_unit_labels:
                    # labels.add(l.lower())
                    labels.add(l)
    labels = list(labels)
    labels = sorted(labels)
    label_dict = {l: idx for idx, l in enumerate(labels)}
    print(label_dict)
    print(labels)

    return label_dict, labels

def seg_preds_to_file(pred_ids, restore_ids, label_list, gold_file):
    """
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        pred_ids:
        restore_ids:
        gold_file:
        label_list:
    """
    pass

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



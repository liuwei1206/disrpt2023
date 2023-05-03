import os
import json
import io

from utils import *


def tok_reader(file_name):
    """
    Args:
        file_name: data path
    """
    all_doc_data = {}
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc = []
        for line in lines:
            line = line.strip()
            if line:
                if "newdoc_id" in line.lower():
                    tmp_doc_id = line.split("=")[1].strip()
                else:
                    items = line.split("\t")  # check if is \t
                    if "-" in items[0]:  # ignore such as 16-17
                        continue
                    token_id = int(items[0].strip())
                    token = items[1].strip()
                    token_label = items[-1].strip()
                    tmp_doc.append((token_id, token, token_label))
            else:
                if len(tmp_doc) > 0 and tmp_doc_id is not None:
                    # all_doc_data.append((tmp_doc_id, tmp_doc))
                    all_doc_data[tmp_doc_id] = tmp_doc
                tmp_doc_id = None
                tmp_doc = []

    # in case the last one
    if len(tmp_doc) > 0 and tmp_doc_id is not None:
        # all_doc_data.append((tmp_doc_id, tmp_doc))
        all_doc_data[tmp_doc_id] = tmp_doc
    return all_doc_data


def conll_reader(file_name):
    all_conll_data = {}
    # errors accumulator for tok id
    tok_id_acc = 0
    # accumulator for token id
    acc_4_id = 0
    # accumulator for the previous tokens in sentences
    acc_4_sent = 0
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc_info = []
        tmp_sent_id = None
        tmp_sent_type = None
        tmp_sent_info = []
        cur_id = 0
        for line in lines:
            line = line.strip()
            if line:
                if "newdoc_id" in line.lower():
                    if len(tmp_doc_info) > 0 and tmp_doc_id is not None:
                        # all_conll_data.append((tmp_doc_id, tmp_doc_info))
                        all_conll_data[tmp_doc_id] = tmp_doc_info
                    # reset
                    tmp_doc_id = line.split("=")[1].strip()
                    tmp_doc_info = []
                    acc_4_id = 0
                    acc_4_sent = 0
                else:
                    if "sent_id" in line.lower() in line.lower() or "newutterance_id" in line.lower() in line.lower():
                        tmp_sent_id = line.split("=")[1].strip()
                        # reset the accumulator for the wrong labeld tok id
                        acc_4_sent += acc_4_id
                        acc_4_id = 0
                        tok_id_acc = 0
                    elif "s_type" in line.lower():
                        tmp_sent_type = line.split("=")[1].strip()
                    elif line.lower()[0].isdigit():
                        # you can read now word information here
                        items = line.split("\t")  # check if is \t
                        token_id = items[0]
                        if "-" in token_id:  # ignore invalid
                            continue
                        # eng.rst.gum_train.conllu has this issue for wrong labeled tok id(the word denoted)
                        elif "." in token_id:
                            tok_id_acc += 1
                        # wrong labeling issue in ./data/ita.pdtb.luna/ita.pdtb.luna_dev.conllu the token id is 14si
                        # the token's id concatenates the token's string...

                        # this if condition is for spa. file, because there's no sent_id!
                        elif "spa." in file_name and cur_id > int(float(token_id)):
                            acc_4_sent += acc_4_id
                            acc_4_id = 0
                            tok_id_acc = 0
                        else:
                            real_id = ""
                            real_string = ""
                            for i in token_id:
                                if i.isdigit():
                                    real_id += i
                                else:
                                    real_string += i
                            token_id = real_id
                            items[0] = token_id
                            items.insert(1, real_string)
                        POS1 = items[3]
                        POS2 = items[4]
                        POS3 = items[5]
                        POS4 = items[6]
                        POS5 = items[7]
                        POS6 = items[8]

                        acc_4_id += 1

                        tmp_sent_info.append(
                            (int(float(token_id)) + tok_id_acc + acc_4_sent, POS1, POS2, POS3, POS4, POS5, POS6))
                        cur_id = int(float(token_id))
                    else:
                        continue
            else:
                if tmp_doc_info is not None:
                    tmp_doc_info.append((tmp_sent_id, tmp_sent_type, tmp_sent_info))
                else:
                    raise Exception("The tmp_doc_info should not be None!!!")
                # reset
                tmp_sent_id = None
                tmp_sent_type = None
                tmp_sent_info = []
        # here, shouldn't check the tmp_sent_id is None or not, because in the file zho.pdtb.cdtb_train.collu, there's no information
        # about the sentence id!
        # if tmp_sent_id is not None:
        if tmp_sent_info is not None:
            if tmp_doc_info is not None:
                tmp_doc_info.append((tmp_sent_id, tmp_sent_type, tmp_sent_info))
            else:
                raise Exception("The tmp_doc_info should not be None!!!")
        if len(tmp_doc_info) > 0:
            all_conll_data[tmp_doc_id] = tmp_doc_info
    return all_conll_data


def rel_reader(file_name):
    all_relation_data = {}
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                items = line.split("\t")
                doc_id = items[0]
                unit1_toks = items[1]
                unit2_toks = items[2]
                # label = items[11]
                label = items[-1]
                if doc_id in all_relation_data:
                    all_relation_data[doc_id].append((unit1_toks, unit2_toks, label))
                else:
                    all_relation_data[doc_id] = [(unit1_toks, unit2_toks, label)]

    return all_relation_data


"""
def preprocessing(tok_file, conllu_file, rel_file, output_file):

    Args:
        tok_file: raw text
        conllu_file: parsed results of the text
        rel_file: relation annotation file

    all_doc_data = tok_reader(tok_file)
    all_conll_data = conll_reader(conllu_file)
    all_relation_data = rel_reader(rel_file)

    assert len(all_doc_data) == len(all_conll_data), (len(all_doc_data), len(all_conll_data))
    # assert len(all_doc_data) == len(all_relation_data), (len(all_doc_data), len(all_relation_data))

    all_samples = []

    for doc, conll in zip(all_doc_data, all_conll_data):
        doc_id = doc[0]
        doc_tokens = doc[1]
        print(conllu_file)
        assert doc_id == conll[0], (doc_id, conll[0])
        doc_conll_info = conll[1]
        flat_doc_conll_info = []

        ## for segmentation and connective detection
        doc_sent_tokens = []
        doc_sent_token_features = []
        doc_sent_token_labels = []
        for sent in doc_conll_info:
            sent_tokens = []
            sent_features = []
            sent_labels = []
            for token_info in sent[2]:
                token_id = int(token_info[0]) # start from 1
                POS1 = token_info[1]
                POS2 = token_info[2]
                token = doc_tokens[token_id-1]
                assert token[0] == token_id, (token[0], token_id)
                sent_tokens.append(token[1])
                sent_features.append((POS1, POS2))
                sent_labels.append(token[2])

            doc_sent_tokens.append(sent_tokens)
            doc_sent_token_features.append(sent_features)
            doc_sent_token_labels.append(sent_labels)
            flat_doc_conll_info.extend(sent[2])
        assert len(doc_tokens) == len(flat_doc_conll_info), (len(doc_tokens), len(flat_doc_conll_info))
        ## for relation classification
        doc_unit_tokens = []
        doc_unit_token_features = []
        doc_unit_labels = []
        if doc_id in all_relation_data:
            doc_rel = all_relation_data[doc_id]
        else:
            doc_rel = []
        for unit_pair in doc_rel:
            unit1_ids = unit_pair[0]
            unit2_ids = unit_pair[1]
            rel = unit_pair[2]
            doc_unit_labels.append(rel)

            unit1_tokens = []
            unit2_tokens = []
            unit1_features = []
            unit2_features = []
            group_unit1_ids = unit1_ids.split(",")
            group_unit2_ids = unit2_ids.split(",")
            for span in group_unit1_ids:
                if "-" in span: # a range
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end+1):
                        unit1_tokens.append(doc_tokens[idx-1][1])
                        unit1_features.append(flat_doc_conll_info[idx-1][1:])
                else: # a number
                    span_pos = int(span)
                    unit1_tokens.append(doc_tokens[span_pos-1][1])
                    unit1_features.append(flat_doc_conll_info[span_pos-1][1:])

            for span in group_unit2_ids:
                # a range
                if "-" in span:
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end+1):
                        unit2_tokens.append(doc_tokens[idx-1][1])
                        unit2_features.append(flat_doc_conll_info[idx-1][1:])
                else: # a number
                    span_pos = int(span)
                    unit2_tokens.append(doc_tokens[span_pos-1][1])
                    unit2_features.append(flat_doc_conll_info[span_pos-1][1:])

            doc_unit_tokens.append((unit1_tokens, unit2_tokens))
            doc_unit_token_features.append((unit1_features, unit2_features))

        # save info json
        sample = {}
        sample["doc_id"] = doc_id
        sample["doc_sents"] = doc_sent_tokens
        sample["doc_sent_token_features"] = doc_sent_token_features
        sample["doc_sent_token_labels"] = doc_sent_token_labels

        sample["doc_units"] = doc_unit_tokens
        sample["doc_unit_token_features"] = doc_unit_token_features
        sample["doc_unit_labels"] = doc_unit_labels

        all_samples.append(json.dumps(sample, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_samples:
            f.write("%s\n"%(text))

"""


def preprocessing(tok_file, conllu_file, rel_file, output_file):
    """
    Args:
        tok_file: raw text
        conllu_file: parsed results of the text
        rel_file: relation annotation file
    """

    all_doc_data = tok_reader(tok_file)
    all_conll_data = conll_reader(conllu_file)
    all_relation_data = rel_reader(rel_file)

    assert len(all_doc_data) == len(all_conll_data), (len(all_doc_data), len(all_conll_data))
    # assert len(all_doc_data) == len(all_relation_data), (len(all_doc_data), len(all_relation_data))

    all_samples = []

    for doc_id in all_doc_data:
        doc_tokens = all_doc_data[doc_id]
        assert doc_id in all_conll_data
        doc_conll_info = all_conll_data[doc_id]
        flat_doc_conll_info = []

        ## for segmentation and connective detection
        doc_sent_tokens = []
        doc_sent_token_features = []
        doc_sent_token_labels = []
        for sent in doc_conll_info:
            sent_tokens = []
            sent_features = []
            sent_labels = []
            for token_info in sent[2]:
                token_id = int(token_info[0])  # start from 1
                POS1 = token_info[1]
                POS2 = token_info[2]
                POS3 = token_info[3]
                POS4 = token_info[4]
                POS5 = token_info[5]
                POS6 = token_info[6]
                token = doc_tokens[token_id - 1]
                assert token[0] == token_id, (token[0], token_id)
                sent_tokens.append(token[1])
                sent_features.append((POS1, POS2, POS3, POS4, POS5, POS6))
                sent_labels.append(token[2])
            doc_sent_tokens.append(sent_tokens)
            doc_sent_token_features.append(sent_features)
            doc_sent_token_labels.append(sent_labels)
            flat_doc_conll_info.extend(sent[2])
        assert len(doc_tokens) == len(flat_doc_conll_info), (len(doc_tokens), len(flat_doc_conll_info))
        ## for relation classification
        doc_unit_tokens = []
        doc_unit_token_features = []
        doc_unit_labels = []

        if doc_id in all_relation_data:
            doc_rel = all_relation_data[doc_id]
        else:
            doc_rel = []

        for unit_pair in doc_rel:
            unit1_ids = unit_pair[0]
            unit2_ids = unit_pair[1]
            rel = unit_pair[2]
            doc_unit_labels.append(rel)

            unit1_tokens = []
            unit2_tokens = []
            unit1_features = []
            unit2_features = []
            group_unit1_ids = unit1_ids.split(",")
            group_unit2_ids = unit2_ids.split(",")
            for span in group_unit1_ids:
                if "-" in span:  # a range
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end + 1):
                        unit1_tokens.append(doc_tokens[idx - 1][1])
                        unit1_features.append(flat_doc_conll_info[idx - 1][1:])
                # bug in ./data/ita.pdtb.luna/ita.pdtb.luna_train.rels
                # you can look at line 9, the number of unit_toks is "_"
                elif "_" in span:  # means no unit_toks
                    unit1_tokens.append("_")
                    unit1_features.append(("_", "_"))
                else:  # a number
                    span_pos = int(span)
                    unit1_tokens.append(doc_tokens[span_pos - 1][1])
                    unit1_features.append(flat_doc_conll_info[span_pos - 1][1:])

            for span in group_unit2_ids:
                # a range
                if "-" in span:
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end + 1):
                        unit2_tokens.append(doc_tokens[idx - 1][1])
                        unit2_features.append(flat_doc_conll_info[idx - 1][1:])
                elif "_" in span:  # means no unit_toks
                    unit2_tokens.append("_")
                    unit2_features.append(("_", "_"))
                else:  # a number
                    span_pos = int(span)
                    unit2_tokens.append(doc_tokens[span_pos - 1][1])
                    unit2_features.append(flat_doc_conll_info[span_pos - 1][1:])

            doc_unit_tokens.append((unit1_tokens, unit2_tokens))
            doc_unit_token_features.append((unit1_features, unit2_features))

        # save info json
        sample = {}
        sample["doc_id"] = doc_id
        sample["doc_sents"] = doc_sent_tokens
        sample["doc_sent_token_features"] = doc_sent_token_features
        sample["doc_sent_token_labels"] = doc_sent_token_labels
        sample["doc_units"] = doc_unit_tokens
        sample["doc_unit_token_features"] = doc_unit_token_features
        sample["doc_unit_labels"] = doc_unit_labels

        all_samples.append(json.dumps(sample, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_samples:
            f.write("%s\n" % (text))


# Because the tur.pdtb.tdb has no .tok file

def conll2tok_reader_tur(file_name):
    """
    Args:
        file_name: data path for Turkish .collu file
    """
    conll_2_tok = {}
    # errors accumulator for tok id
    tok_id_acc = 0
    # accumulator for token id
    acc_4_id = 0
    # accumulator for the previous tokens in sentences
    acc_4_sent = 0
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc_info = []
        for line in lines:
            line = line.strip()
            if line:
                if "newdoc_id" in line.lower():
                    if len(tmp_doc_info) > 0 and tmp_doc_id is not None:
                        # all_conll_data.append((tmp_doc_id, tmp_doc_info))
                        conll_2_tok[tmp_doc_id] = tmp_doc_info
                    # reset
                    tmp_doc_id = line.split("=")[1].strip()
                    tmp_doc_info = []
                    acc_4_id = 0
                    acc_4_sent = 0
                elif "sent_id" in line.lower():
                    acc_4_sent += acc_4_id
                    acc_4_id = 0
                    tok_id_acc = 0
                else:
                    if line.lower()[0].isdigit():
                        # you can read now word information here
                        items = line.split("\t")  # check if is \t
                        token_id = items[0]
                        if "-" in token_id:  # ignore invalid
                            continue
                        # eng.rst.gum_train.conllu has this issue for wrong labeled tok id(the word denoted)
                        elif "." in token_id:
                            tok_id_acc += 1
                        # wrong labeling issue in ./data/ita.pdtb.luna/ita.pdtb.luna_dev.conllu the token id is 14si
                        # the token's id concatenates the token's string...
                        else:
                            real_id = ""
                            real_string = ""
                            for i in token_id:
                                if i.isdigit():
                                    real_id += i
                                else:
                                    real_string += i
                            token_id = real_id
                            items[0] = token_id
                            items.insert(1, real_string)
                        # the word
                        POS1 = items[2]
                        # suppose to be the segmentation sign, but in tur, it has no such a label
                        POS2 = "_"
                        acc_4_id += 1
                        token_id = str(int(token_id) + acc_4_sent)
                        tmp_doc_info.append((int(float(token_id)) + tok_id_acc, POS1, POS2))
                    else:
                        continue
        # if tmp_sent_id is not None:
        if len(tmp_doc_info) > 0:
            conll_2_tok[tmp_doc_id] = tmp_doc_info
    return conll_2_tok


def convert_tur(conllu_file, rel_file, output_file):
    """
    Args:
        conllu_file: data path for Turkish .collu file
        rel_file: data path for Turkish .rel file
        output_file: data path for the output json file
    """
    print("++++++++++++++")
    all_doc_data = conll2tok_reader_tur(conllu_file)
    print("+++++++++++++++++++++++++++++")
    all_conll_data = conll_reader(conllu_file)
    print("+++++++++++++++++++++++++++++++++++++++")
    all_relation_data = rel_reader(rel_file)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

    assert len(all_doc_data) == len(all_conll_data), (len(all_doc_data), len(all_conll_data))
    # assert len(all_doc_data) == len(all_relation_data), (len(all_doc_data), len(all_relation_data))

    all_samples = []

    for doc_id in all_doc_data:
        doc_tokens = all_doc_data[doc_id]
        # print(conllu_file)
        assert doc_id in all_conll_data
        doc_conll_info = all_conll_data[doc_id]
        flat_doc_conll_info = []

        ## for segmentation and connective detection
        doc_sent_tokens = []
        doc_sent_token_features = []
        doc_sent_token_labels = []
        for sent in doc_conll_info:
            sent_tokens = []
            sent_features = []
            sent_labels = []
            for token_info in sent[2]:
                token_id = int(token_info[0])  # start from 1
                POS1 = token_info[1]
                POS2 = token_info[2]
                token = doc_tokens[token_id - 1]
                assert token[0] == token_id, (token[0], token_id)
                sent_tokens.append(token[1])
                sent_features.append((POS1, POS2))
                sent_labels.append(token[2])

            doc_sent_tokens.append(sent_tokens)
            doc_sent_token_features.append(sent_features)
            doc_sent_token_labels.append(sent_labels)
            flat_doc_conll_info.extend(sent[2])
        assert len(doc_tokens) == len(flat_doc_conll_info), (len(doc_tokens), len(flat_doc_conll_info))
        ## for relation classification
        doc_unit_tokens = []
        doc_unit_token_features = []
        doc_unit_labels = []
        if doc_id in all_relation_data:
            doc_rel = all_relation_data[doc_id]
        else:
            doc_rel = []
        for unit_pair in doc_rel:
            unit1_ids = unit_pair[0]
            unit2_ids = unit_pair[1]
            rel = unit_pair[2]
            doc_unit_labels.append(rel)

            unit1_tokens = []
            unit2_tokens = []
            unit1_features = []
            unit2_features = []
            group_unit1_ids = unit1_ids.split(",")
            group_unit2_ids = unit2_ids.split(",")
            for span in group_unit1_ids:
                if "-" in span:  # a range
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end + 1):
                        unit1_tokens.append(doc_tokens[idx - 1][1])
                        unit1_features.append(flat_doc_conll_info[idx - 1][1:])
                # bug in ./data/ita.pdtb.luna/ita.pdtb.luna_train.rels
                # you can look at line 9, the number of unit_toks is "_"
                elif "_" in span:  # means no unit_toks
                    unit1_tokens.append("_")
                    unit1_features.append(("_", "_"))
                else:  # a number
                    span_pos = int(span)
                    unit1_tokens.append(doc_tokens[span_pos - 1][1])
                    unit1_features.append(flat_doc_conll_info[span_pos - 1][1:])

            for span in group_unit2_ids:
                # a range
                if "-" in span:
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end + 1):
                        unit2_tokens.append(doc_tokens[idx - 1][1])
                        unit2_features.append(flat_doc_conll_info[idx - 1][1:])
                elif "_" in span:  # means no unit_toks
                    unit2_tokens.append("_")
                    unit2_features.append(("_", "_"))
                else:  # a number
                    span_pos = int(span)
                    unit2_tokens.append(doc_tokens[span_pos - 1][1])
                    unit2_features.append(flat_doc_conll_info[span_pos - 1][1:])

            doc_unit_tokens.append((unit1_tokens, unit2_tokens))
            doc_unit_token_features.append((unit1_features, unit2_features))

        # save info json
        sample = {}
        sample["doc_id"] = doc_id
        sample["doc_sents"] = doc_sent_tokens
        sample["doc_sent_token_features"] = doc_sent_token_features
        sample["doc_sent_token_labels"] = doc_sent_token_labels

        sample["doc_units"] = doc_unit_tokens
        sample["doc_unit_token_features"] = doc_unit_token_features
        sample["doc_unit_labels"] = doc_unit_labels

        all_samples.append(json.dumps(sample, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_samples:
            f.write("%s\n" % (text))


def convert_all(data_folder_path):
    folder_names = os.listdir(data_folder_path)
    for names in folder_names:
        print(names)
        # read all training files in one folder
        train_tok_file = data_folder_path + names + "/" + names + "_train.tok"
        train_conllu_file = data_folder_path + names + "/" + names + "_train.conllu"
        train_rels_file = data_folder_path + names + "/" + names + "_train.rels"
        output_file_train = data_folder_path + names + "/" + names + "_train.json"
        if os.path.exists(train_tok_file):
            preprocessing(train_tok_file, train_conllu_file, train_rels_file, output_file_train)

        # read all development files in one folder
        dev_tok_file = data_folder_path + names + "/" + names + "_dev.tok"
        dev_conllu_file = data_folder_path + names + "/" + names + "_dev.conllu"
        dev_rels_file = data_folder_path + names + "/" + names + "_dev.rels"
        output_file_dev = data_folder_path + names + "/" + names + "_dev.json"
        if os.path.exists(dev_tok_file):
            preprocessing(dev_tok_file, dev_conllu_file, dev_rels_file, output_file_dev)

        # read all test files in one folder
        test_tok_file = data_folder_path + names + "/" + names + "_test.tok"
        test_conllu_file = data_folder_path + names + "/" + names + "_test.conllu"
        test_rels_file = data_folder_path + names + "/" + names + "_test.rels"
        output_file_test = data_folder_path + names + "/" + names + "_test.json"
        if os.path.exists(test_tok_file):
            preprocessing(test_tok_file, test_conllu_file, test_rels_file, output_file_test)

if __name__ == "__main__":
    convert_all("data/dataset/")

    """
    # turkish
    tur_train_conll_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_train.conllu"
    tur_train_rel_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_train.rels"
    tur_train_output_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_train.json"
    convert_tur(tur_train_conll_file, tur_train_rel_file, tur_train_output_file)

    tur_dev_conll_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_dev.conllu"
    tur_dev_rel_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_dev.rels"
    tur_dev_output_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_dev.json"
    convert_tur(tur_dev_conll_file, tur_dev_rel_file, tur_dev_output_file)

    tur_test_conll_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_test.conllu"
    tur_test_rel_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_test.rels"
    tur_test_output_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_test.json"
    convert_tur(tur_test_conll_file, tur_test_rel_file, tur_test_output_file)
    """
    
    """
    # zho
    dataset = "deu.rst.pcc"
    for mode in ["train", "dev", "test"]:
        tok_file = "data/dataset/{}/{}_{}.tok".format(dataset, dataset, mode)
        conll_file = "data/dataset/{}/{}_{}.conllu".format(dataset, dataset, mode)
        rel_file = "data/dataset/{}/{}_{}.rels".format(dataset, dataset, mode)
        out_file = "data/dataset/{}/{}_{}.json".format(dataset, dataset, mode)
        preprocessing(tok_file, conll_file, rel_file, out_file)

    """
    # This part of code is for generating one dataset's fasttext embedding, everytime when you want to
    # generate the fasttext embedding from one dataset, please download it first!

    #missing fasttext dictionary for eng.pdtb.pdtb, eng.rst.rstdt, tur.pdtb.tdb, tur.pdtb.tedm, zho.pdtb.cdtb

    # choose one dataset that you want to generate fasttext dictionary
    dataset_path = "data/dataset/"
    ft_target_dataset = "deu.rst.pcc"
    ft_train_path = dataset_path + ft_target_dataset + "/" + ft_target_dataset + "_train.json"
    ft_dev_path = dataset_path + ft_target_dataset + "/" + ft_target_dataset + "_dev.json"
    ft_test_path = dataset_path + ft_target_dataset + "/" + ft_target_dataset + "_test.json"
    ft_output_path = dataset_path + ft_target_dataset + "/" + ft_target_dataset + "_ftdict.npy"

    ft_target_language = ft_target_dataset.split(".")[0]
    print(ft_target_language)

    """
    if ft_target_language.lower() == "deu":
        fasttext_language = "de"
        fasttext_model = "cc.de.300.bin"

    elif ft_target_language.lower() == "eng":
        fasttext_language = "en"
        fasttext_model = "cc.en.300.bin"

    elif ft_target_language.lower() == "eus":
        fasttext_language = "eu"
        fasttext_model = "cc.eu.300.bin"

    elif ft_target_language.lower() == "fas":
        fasttext_language = "fa"
        fasttext_model = "cc.fa.300.bin"

    elif ft_target_language.lower() == "fra":
        fasttext_language = "fr"
        fasttext_model = "cc.fr.300.bin"

    elif ft_target_language.lower() == "ita":
        fasttext_language = "it"
        fasttext_model = "cc.it.300.bin"

    elif ft_target_language.lower() == "nld":
        fasttext_language = "nl"
        fasttext_model = "cc.nl.300.bin"

    elif ft_target_language.lower() == "por":
        fasttext_language = "pt"
        fasttext_model = "cc.pt.300.bin"

    elif ft_target_language.lower() == "rus":
        fasttext_language = "ru"
        fasttext_model = "cc.ru.300.bin"

    elif ft_target_language.lower() == "spa":
        fasttext_language = "es"
        fasttext_model = "cc.es.300.bin"

    elif ft_target_language.lower() == "tur":
        fasttext_language = "tr"
        fasttext_model = "cc.tr.300.bin"

    elif ft_target_language.lower() == "zho":
        fasttext_language = "zh"
        fasttext_model = "cc.zh.300.bin"

    elif ft_target_language.lower() == "tha":
        fasttext_language = "th"
        fasttext_model = "cc.th.300.bin"

    # generate_ft_dict(ft_train_path, ft_dev_path, ft_test_path, ft_output_path, "data/dataset/" + fasttext_model, fasttext_language)
    generate_ft_dict(ft_train_path, ft_dev_path, ft_test_path, ft_output_path, "data/embeddings/" + fasttext_model, fasttext_language)
    #ft_dict = np.load(ft_output_path, allow_pickle=True).item()
    #print(ft_dict["посте"])
    """

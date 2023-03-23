
import os
import json

def tok_reader(file_name):
    """
    Args:
        file_name: data path
    """
    all_doc_data = []
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
                    items = line.split("\t") # check if is \t
                    if "-" in items[0]: # ignore such as 16-17
                        continue
                    token_id = int(items[0].strip())
                    token = items[1].strip()
                    token_label = items[9].strip()
                    tmp_doc.append((token_id, token, token_label))
            else:
                if len(tmp_doc) > 0 and tmp_doc_id is not None:
                    all_doc_data.append((tmp_doc_id, tmp_doc))
                tmp_doc_id = None
                tmp_doc = []

    # in case the last one
    if len(tmp_doc) > 0 and tmp_doc_id is not None:
        all_doc_data.append((tmp_doc_id, tmp_doc))

    return all_doc_data

def conll_reader(file_name):
    all_conll_data = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc_info = []
        tmp_sent_id = None
        tmp_sent_type = None
        tmp_sent_info = []
        for line in lines:
            line = line.strip()
            if line:
                if "newdoc_id" in line.lower():
                    if len(tmp_doc_info) > 0 and tmp_doc_id is not None:
                        all_conll_data.append((tmp_doc_id, tmp_doc_info))
                    # reset
                    tmp_doc_id = line.split("=")[1].strip()
                    tmp_doc_info = []
                else:
                    if "sent_id" in line.lower():
                        tmp_sent_id = line.split("=")[1].strip()
                    elif "s_type" in line.lower():
                        tmp_sent_type = line.split("=")[1].strip()
                    elif "# text" in line.lower():
                        continue
                    else:
                        # you can read now word information here
                        items = line.split("\t") # check if is \t
                        token_id = items[0]
                        if "-" in token_id: # ignore invalid
                            continue
                        POS1 = items[3]
                        POS2 = items[4]

                        tmp_sent_info.append((token_id, POS1, POS2))
            else:
                if tmp_doc_info is not None:
                    tmp_doc_info.append((tmp_sent_id, tmp_sent_type, tmp_sent_info))
                else:
                    raise Exception("The tmp_doc_info should not be None!!!")
                # reset
                tmp_sent_id = None
                tmp_sent_type = None
                tmp_sent_info = []
        if tmp_sent_id is not None:
            if tmp_doc_info is not None:
                tmp_doc_info.append((tmp_sent_id, tmp_sent_type, tmp_sent_info))
            else:
                raise Exception("The tmp_doc_info should not be None!!!")
        if len(tmp_doc_info) > 0:
            all_conll_data.append((tmp_doc_id, tmp_doc_info))

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
                label = items[11]
                if doc_id in all_relation_data:
                    all_relation_data[doc_id].append((unit1_toks, unit2_toks, label))
                else:
                    all_relation_data[doc_id] = [(unit1_toks, unit2_toks, label)]

    return all_relation_data


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
    for doc, conll in zip(all_doc_data, all_conll_data):
        doc_id = doc[0]
        doc_tokens = doc[1]
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


if __name__ == "__main__":
    tok_file = "data/eng.pdtb.pdtb/eng.pdtb.pdtb_dev.tok"
    conll_file = "data/eng.pdtb.pdtb/eng.pdtb.pdtb_dev.conllu"
    rel_file = "data/eng.pdtb.pdtb/eng.pdtb.pdtb_dev.rels"
    output_file = "data/eng.pdtb.pdtb/dev.json"

    preprocessing(tok_file, conll_file, rel_file, output_file)


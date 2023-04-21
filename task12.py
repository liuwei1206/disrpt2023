import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta import RobertaConfig, RobertaTokenizer
from transformers.models.bert import BertConfig, BertTokenizer
from transformers.models.electra import ElectraConfig, ElectraTokenizer
# from transformers.models.xlm_roberta import XLMRobertaConfig, XLMRobertaTokenizer
from transformers import XLMRobertaConfig, XLMRobertaTokenizer

from utils import *
from task_dataset import SegDataset
from models import *
from seg_eval import get_scores

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

import warnings
warnings.filterwarnings('ignore')

def get_argparse():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="pdtb2", type=str, help="pdtb2, pdtb3")
    parser.add_argument("--output_dir", default="data/result", type=str)
    parser.add_argument("--feature_size", default=0, type=int)

    # for training
    parser.add_argument("--model_type", default="base", type=str, help="roberta-bilstm-crf")
    parser.add_argument("--encoder_type", default="roberta", type=str, help="roberta, ...")
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--do_freeze", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode="train"):
    print("  {} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
        # here, if you want to use random sampler, then we cannot map the result one-by-one in the evaluate step
        # but, since it's training stage, I think it's ok, you can input the training file again under the test stage.
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader


def get_optimizer(model, args, num_training_steps):
    specific_params = []
    no_deday = ["bias", "LayerNorm.weigh"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train(model, args, tokenizer, train_dataloader, dev_dataloader=None, test_dataloader=None):
    # 1.prepare
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    print_step = int(len(train_dataloader) // 4) + 1
    num_train_epochs = args.num_train_epochs
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info(" ***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    # 2.train
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Train"
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            logging_loss = loss.item() * args.train_batch_size
            tr_loss += logging_loss

            if global_step % print_step == 0:
                print(" global_step=%d, cur loss=%.4f, global avg loss=%.4f" % (
                        global_step, logging_loss, tr_loss / global_step)
                )

        # 3. evaluate and save
        model.eval()
        if False and train_dataloader is not None:
            score_dict = evaluate(model, args, train_dataloader, tokenizer, epoch, desc="train")
            print("\nTrain: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        if dev_dataloader is not None:
            score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            if score_dict["f_score"] > best_dev:
                best_dev = score_dict["f_score"]
            print("\nDev: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        if test_dataloader is not None:
            score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
            print("\nTest: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print("\nBest F1 on dev: %.4f"%(best_dev))

def train_plus(model, args, tokenizer, train_dataloader, dev_dataloader=None, test_dataloader=None, extra_feat_len=None):
    # 1.prepare
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    print_step = int(len(train_dataloader) // 4) + 1
    num_train_epochs = args.num_train_epochs
    optimizer, scheduler = get_optimizer(model, args, t_total)

    logger.info(" ***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    # 2.train
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Train",
                "extra_feat": batch[4]
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            logging_loss = loss.item() * args.train_batch_size
            tr_loss += logging_loss

            if global_step % print_step == 0:
                print(" global_step=%d, cur loss=%.4f, global avg loss=%.4f" % (
                        global_step, logging_loss, tr_loss / global_step)
                )

        # 3. evaluate and save
        model.eval()
        if False and train_dataloader is not None:
            score_dict = evaluate(model, args, train_dataloader, tokenizer, epoch, desc="train")
            print("\nTrain: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        if dev_dataloader is not None:
            score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            if score_dict["f_score"] > best_dev:
                best_dev = score_dict["f_score"]
            print("\nDev: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        if test_dataloader is not None:
            score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
            print("\nTest: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print("\nBest F1 on dev: %.4f"%(best_dev))

def evaluate(model, args, dataloader, tokenizer, epoch, desc="dev", write_file=False):
    all_input_ids = None
    all_attention_mask = None
    all_label_ids = None
    all_pred_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_input_ids is None:
            all_input_ids = input_ids
            all_attention_mask = attention_mask
            all_label_ids = label_ids
            all_pred_ids = pred_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_pred_ids = np.append(all_pred_ids, pred_ids, axis=0)

    ## evaluation
    if desc == "train":
        gold_file = args.train_data_file.replace(".json", ".tok")
    elif desc == "dev":
        gold_file = args.dev_data_file.replace(".json", ".tok")
    elif desc == "test":
        gold_file = args.test_data_file.replace(".json", ".tok")
    print(all_pred_ids)
    pred_file = seg_preds_to_file(all_input_ids, all_pred_ids, all_attention_mask, args.tokenizer, args.label_list, gold_file)
    score_dict = get_scores(gold_file, pred_file)

    return score_dict


def evaluate_new(model, args, dataloader, tokenizer, epoch, desc="dev", write_file=False):
    all_input_ids = None
    all_attention_mask = None
    all_label_ids = None
    all_pred_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        og_tok_idxs = batch[3].detach().cpu().numpy()
        if all_input_ids is None:
            all_input_ids = input_ids
            all_attention_mask = attention_mask
            all_label_ids = label_ids
            all_pred_ids = pred_ids
            all_tok_idxs = og_tok_idxs
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_pred_ids = np.append(all_pred_ids, pred_ids, axis=0)
            all_tok_idxs = np.append(all_tok_idxs, og_tok_idxs, axis=0)
    ## evaluation
    if desc == "train":
        gold_file = args.train_data_file.replace(".json", ".tok")
    elif desc == "dev":
        gold_file = args.dev_data_file.replace(".json", ".tok")
    elif desc == "test":
        gold_file = args.test_data_file.replace(".json", ".tok")
    print(all_pred_ids)
    pred_file = seg_preds_to_file_new(all_input_ids, all_label_ids, all_attention_mask, all_tok_idxs, args.tokenizer, args.label_list, gold_file)
    score_dict = get_scores(gold_file, pred_file)

    return score_dict

def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    # 1.prepare pretrained path
    lang_type = args.dataset.split(".")[0]
    args.lang_type = lang_type
    print(lang_type)
    if lang_type.lower() == "deu":
        # encoder_type = "xlm-roberta"
        # pretrained_path = "xlm-roberta-large"
        encoder_type = "bert"
        pretrained_path = "bert-base-german-cased"
    elif lang_type.lower() == "eng":
        encoder_type = "bert" # "electra"
        pretrained_path = "bert-base-cased" # "google/electra-large-discriminator"
    elif lang_type.lower() == "eus":
        encoder_type = "bert"
        pretrained_path = "ixa-ehu/berteus-base-cased"
    elif lang_type.lower() == "fas":
        encoder_type = "bert"
        pretrained_path = "HooshvareLab/bert-fa-base-uncased"
    elif lang_type.lower() == "fra":
        encoder_type = "xlm-roberta"
        pretrained_path = "xlm-roberta-large"
    elif lang_type.lower() == "ita":
        encoder_type = "bert"
        pretrained_path = "dbmdz/bert-base-italian-cased"
    elif lang_type.lower() == "nld":
        encoder_type = "roberta"
        pretrained_path = "pdelobelle/robbert-v2-dutch-base"
    elif lang_type.lower() == "por":
        encoder_type = "bert"
        pretrained_path = "neuralmind/bert-base-portuguese-cased"
    elif lang_type.lower() == "rus":
        encoder_type = "bert"
        pretrained_path = "DeepPavlov/rubert-base-cased"
    elif lang_type.lower() == "spa":
        encoder_type = "bert"
        pretrained_path = "dccuchile/bert-base-spanish-wwm-cased"
    elif lang_type.lower() == "tur":
        encoder_type = "bert"
        pretrained_path = "dbmdz/bert-base-turkish-cased"
    elif lang_type.lower() == "zho":
        encoder_type = "bert"
        pretrained_path = "bert-base-chinese"
    args.encoder_type = encoder_type
    args.pretrained_path = pretrained_path

    # 2.prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    args.data_dir = data_dir
    train_data_file = os.path.join(data_dir, "{}_train.json".format(args.dataset))
    dev_data_file = os.path.join(data_dir, "{}_dev.json".format(args.dataset))
    test_data_file = os.path.join(data_dir, "{}_test.json".format(args.dataset))
    label_dict, label_list = token_labels_from_file(train_data_file)
    tok_pos_1, tok_pos_2, tok_pos_1_dict, tok_pos_2_dict = token_pos_from_file(train_data_file)
    args.train_data_file, args.dev_data_file, args.test_data_file = train_data_file, dev_data_file, test_data_file
    args.label_dict, args.label_list, args.num_labels = label_dict, label_list, len(label_list)

    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "{}+{}".format(args.model_type, args.encoder_type))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    # 2.define models
    if args.model_type.lower() == "base":
        if args.encoder_type.lower() == "roberta":
            config = RobertaConfig.from_pretrained(pretrained_path)
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "bert":
            config = BertConfig.from_pretrained(pretrained_path)
            tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "electra":
            config = ElectraConfig.from_pretrained(pretrained_path)
            tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "xlm-roberta":
            config = XLMRobertaConfig.from_pretrained(pretrained_path)
            tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
        model = BaseSegClassifier(config=config, args=args)
        dataset_name = "SegDataset"
    elif args.model_type.lower() == "bilstm+crf":
        if args.encoder_type.lower() == "roberta":
            config = RobertaConfig.from_pretrained(pretrained_path)
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "bert":
            config = BertConfig.from_pretrained(pretrained_path)
            tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "electra":
            config = ElectraConfig.from_pretrained(pretrained_path)
            tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "xlm-roberta":
            config = XLMRobertaConfig.from_pretrained(pretrained_path)
            tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
        model = BiLSTMCRF(config=config, args=args)
        dataset_name = "SegDataset"
        # you can test my new dataset by using folowing code
        # dataset_name = "SegDataset2"
        # now you can aplly SegDatasetPlus
    model = model.to(args.device)
    args.tokenizer = tokenizer

    if args.run_plus:
        dataset_params = {
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
            "label_dict": label_dict,
            "pos1_dict": tok_pos_1_dict,
            "pos1_list": tok_pos_1,
            "pos1_convert": args.pos1_convert,
            "pos2_dict": tok_pos_2_dict,
            "pos2_list": tok_pos_2,
            "pos2_convert": args.pos2_convert,
        }
        dataset_module = __import__("task_dataset")
        MyDataset = getattr(dataset_module, dataset_name)
        extra_feat_len = MyDataset.get_extra_feat_len()
        args.extra_feat_dim = extra_feat_len
    else:
        dataset_params = {
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
            "label_dict": label_dict,
        }
        dataset_module = __import__("task_dataset")
        MyDataset = getattr(dataset_module, dataset_name)

    if args.do_train:
        train_dataset = MyDataset(train_data_file, params=dataset_params)
        dev_dataset = MyDataset(dev_data_file, params=dataset_params)
        print(test_data_file)
        if os.path.exists(test_data_file):
            print("++in++")
            test_dataset = MyDataset(test_data_file, params=dataset_params)
        else:
            test_dataset = None
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        if test_dataset is not None:
            test_dataloader = get_dataloader(test_dataset, args, mode="test")
        else:
            test_dataloader = None
        if args.run_plus:
            train_plus(model, args, tokenizer, train_dataloader, dev_dataloader, test_dataloader, extra_feat_len)
        else:
            train(model, args, tokenizer, train_dataloader, dev_dataloader, test_dataloader)


    if args.do_dev or args.do_test:
        time_dir = "good"
        temp_dir = os.path.join(args.output_dir, time_dir)
        temp_file = os.path.join(temp_dir, "checkpoint_{}/pytorch_model.bin")
        if do_dev:
            dev_dataset = MyDataset(dev_data_file, params=dataset_params)
            dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        if do_test:
            test_dataset = MyDataset(test_data_file, params=dataset_params)
            test_dataloader = get_dataloader(test_dataset, args, mode="test")

        for epoch in range(1, args.num_train_epochs+1):
            checkpoint_file = temp_file.format(str(epoch))
            print(" Epoch: {}".format(str(epoch)))
            print(checkpoint_file)
            args.output_dir = os.path.dirname(checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.do_dev:
                evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev", write_file=True)
                # print(" Dev: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (acc, p, r, f1))
            if args.do_test:
                evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test", write_file=True)
                # print(" Test: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (acc, p, r, f1))
            print()

if __name__ == "__main__":
    main()

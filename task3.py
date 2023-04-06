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
from transformers import RobertaConfig, RobertaTokenizer

from utils import *
from task_dataset import SegDataset
from models import *
from rel_eval import get_accuracy_score

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
    parser.add_argument("--learning_rate", default=3e-3, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.0, type=float)
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
    print_step = int(len(train_dataloader) // 4)
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
        if dev_dataloader is not None:
            score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            print(" Dev: Epoch=%d, Acc=%.4f\n" % (epoch, score_dict["acc_score"]))
        if test_dataloader is not None:
            score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
            print(" Test: Epoch=%d, Acc=%.4f\n" % (epoch, score_dict["acc_score"]))
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

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
    if mode == "train":
        gold_file = args.train_data_file.replace(".json", ".rel")
    elif mode == "dev":
        gold_file = args.dev_data_file.replace(".json", ".rel")
    elif mode == "test":
        gold_file = args.test_data_file.replace(".json", ".rel")
    pred_file = rel_preds_to_file(all_pred_ids, args.label_list, gold_file)
    score_dict = get_accuracy_score(gold_file, pred_file)

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

    # 1.prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    args.data_dir = data_dir
    train_data_file = os.path.join(data_dir, "{}_train.json".format(args.dataset))
    dev_data_file = os.path.join(data_dir, "{}_dev.json".format(args.dataset))
    test_data_file = os.path.join(data_dir, "{}_test.json".format(args.dataset))
    label_dict, label_list = token_labels_from_file(train_data_file)

    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "{}+{}".format(args.model_type, args.encoder_type))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    # 2.define models
    if args.model_type.lower() == "base":
        if args.encoder_type.lower() == "roberta":
            config = RobertaConfig.from_pretrained("roberta-base")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = BaseRelClassifier(config=config, args=args)
            dataset_name = "RelDataset"

    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_dict": label_dict,
    }
    dataset_module == __import__("task_dataset")
    MyDataset = getattr(dataset_module, dataset_name)

    if args.do_train:
        train_dataset = MyDataset(train_data_file, params=dataset_params)
        dev_dataset = MyDataset(dev_data_file, params=dataset_params)
        if os.path.exists(test_data_file):
            test_dataset = None
        else:
            test_dataset = MyDataset(test_data_file, params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        if test_dataset is not None:
            test_dataloader = get_dataloader(test_dataset, args, mode="test")
        else:
            test_dataloader = None
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
                score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
                print(" Dev: Epoch=%d, Acc=%.4f\n" % (epoch, score_dict["acc_score"]))
            if args.do_test:
                score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
                print(" Test: Epoch=%d, Acc=%.4f\n" % (epoch, score_dict["acc_score"]))
            print()

if __name__ == "__main__":
    main()
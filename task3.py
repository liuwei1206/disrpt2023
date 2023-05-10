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
from transformers.models.roberta import RobertaConfig, RobertaTokenizer, RobertaModel
from transformers.models.bert import BertConfig, BertTokenizer, BertModel
from transformers.models.electra import ElectraConfig, ElectraTokenizer, ElectraModel
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from transformers import CamembertConfig, CamembertTokenizer, CamembertModel
from sklearn.metrics import f1_score, accuracy_score

from utils import *
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
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--do_freeze", default=False, action="store_true")
    parser.add_argument("--do_adv", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int, help="move to data/config/rel_config.json")
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="move to data/config/rel_config.json")
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
                "token_type_ids": batch[2],
                "labels": batch[3],
                "flag": "Train"
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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
            print("\nTrain: Epoch=%d, Acc=%.4f, F1=%.4f\n" % (epoch, score_dict["acc_score"], score_dict["f1_score"]))
        if dev_dataloader is not None:
            score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            if best_dev < score_dict["acc_score"]:
                best_dev = score_dict["acc_score"] #  + score_dict["f1_score"]
            print("\nDev: Epoch=%d, Acc=%.4f, F1=%.4f\n" % (epoch, score_dict["acc_score"], score_dict["f1_score"]))
        if test_dataloader is not None:
            score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
            print("\nTest: Epoch=%d, Acc=%.4f, F1=%.4f\n" % (epoch, score_dict["acc_score"], score_dict["f1_score"]))
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print("Best Acc on dev: %.4f\n"%(best_dev))


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
            "token_type_ids": batch[2],
            "labels": batch[3],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[3].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()

        if all_input_ids is None:
            all_input_ids = input_ids
            all_attention_mask = attention_mask
            all_label_ids = label_ids
            all_pred_ids = pred_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_pred_ids = np.append(all_pred_ids, pred_ids)

    ## evaluation
    """
    if desc == "train":
        gold_file = args.train_data_file.replace(".json", ".rels")
    elif desc == "dev":
        gold_file = args.dev_data_file.replace(".json", ".rels")
    elif desc == "test":
        gold_file = args.test_data_file.replace(".json", ".rels")
    pred_file = rel_preds_to_file(all_pred_ids, args.label_list, gold_file)
    score_dict = get_accuracy_score(gold_file, pred_file)
    """

    acc = accuracy_score(y_true=all_label_ids, y_pred=all_pred_ids)
    f1 = f1_score(y_true=all_label_ids, y_pred=all_pred_ids, average="macro")
    score_dict = {"acc_score": acc, "f1_score": f1}

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

    # 1. prepare pretrained path
    print("Acc on %s"%(args.dataset))
    rel_config = json.load(open("data/config/rel_config.json"))
    encoder_type = rel_config[args.dataset]["encoder_type"]
    pretrained_path = rel_config[args.dataset]["pretrained_path"]
    args.learning_rate = rel_config[args.dataset]["lr"]
    args.train_batch_size = rel_config[args.dataset]["batch_size"]
    print(" encoder: {}, lr: {}, batch: {}".format(encoder_type, args.learning_rate, args.train_batch_size))
    pretrained_path = os.path.join("/hits/basement/nlp/liuwi/resources/pretrained_models", pretrained_path)
    print(pretrained_path)
    args.encoder_type = encoder_type
    args.pretrained_path = pretrained_path

    # 2.prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    args.data_dir = data_dir
    train_data_file = os.path.join(data_dir, "{}_train.json".format(args.dataset))
    dev_data_file = os.path.join(data_dir, "{}_dev.json".format(args.dataset))
    test_data_file = os.path.join(data_dir, "{}_test.json".format(args.dataset))
    label_dict, label_list = rel_labels_from_file(train_data_file)
    args.train_data_file, args.dev_data_file, args.test_data_file = train_data_file, dev_data_file, test_data_file
    args.label_dict, args.label_list, args.num_labels = label_dict, label_list, len(label_list)

    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "{}+{}".format(args.model_type, args.encoder_type))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    # 3.define models
    if args.model_type.lower() == "base":
        if args.encoder_type.lower() == "roberta":
            config = RobertaConfig.from_pretrained(args.pretrained_path)
            tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_path)
        elif args.encoder_type.lower() == "bert":
            config = BertConfig.from_pretrained(args.pretrained_path)
            tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)
        elif args.encoder_type.lower() == "electra":
            config = ElectraConfig.from_pretrained(pretrained_path)
            tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "xlm-roberta":
            config = XLMRobertaConfig.from_pretrained(pretrained_path)
            tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "camembert":
            config = CamembertConfig.from_pretrained(pretrained_path)
            tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)
        model = BaseRelClassifier(config=config, args=args)
        dataset_name = "RelDataset"
    elif args.model_type.lower() == "multi-task":
        config = XLMRobertaConfig.from_pretrained(pretrained_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
        model = BaseRelClassifier(config=config, args=args)
        dataset_name = "MixedRelDataset"

    model = model.to(args.device)
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
        if os.path.exists(test_data_file):
            test_dataset = MyDataset(test_data_file, params=dataset_params)
        else:
            test_dataset = None
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        if test_dataset is not None:
            test_dataloader = get_dataloader(test_dataset, args, mode="test")
        else:
            test_dataloader = None
        train(model, args, tokenizer, train_dataloader, dev_dataloader, test_dataloader)

    if args.do_dev or args.do_test:
        time_dir = "base"
        temp_dir = os.path.join(args.output_dir, time_dir)
        temp_file = os.path.join(temp_dir, "checkpoint_{}/pytorch_model.bin")
        if args.do_dev:
            ## rst
            # dev_data_file = "data/dataset/eng.rst.gum/eng.rst.gum_dev.json"
            # dev_data_file = "data/dataset/eng.rst.rstdt/eng.rst.rstdt_dev.json"
            # dev_data_file = "data/dataset/eus.rst.ert/eus.rst.ert_dev.json"
            # dev_data_file = "data/dataset/fas.rst.prstc/fas.rst.prstc_dev.json"
            # dev_data_file = "data/dataset/nld.rst.nldt/nld.rst.nldt_dev.json"
            # dev_data_file = "data/dataset/por.rst.cstn/por.rst.cstn_dev.json"
            # dev_data_file = "data/dataset/rus.rst.rrt/rus.rst.rrt_dev.json"
            # dev_data_file = "data/dataset/spa.rst.rststb/spa.rst.rststb_dev.json"
            # dev_data_file = "data/dataset/spa.rst.sctb/spa.rst.sctb_dev.json"
            # dev_data_file = "data/dataset/zho.rst.sctb/zho.rst.sctb_dev.json" 
            ## dep
            # dev_data_file = "data/dataset/eng.dep.scidtb/eng.dep.scidtb_dev.json"
            ## pdtb
            # dev_data_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_dev.json"
            # dev_data_file = "data/dataset/tha.pdtb.tdtb/tha.pdtb.tdtb_dev.json"
            # dev_data_file = "data/dataset/eng.pdtb.pdtb/eng.pdtb.pdtb_dev.json"
            ## sdrt
            dev_data_file = "data/dataset/eng.sdrt.stac/eng.sdrt.stac_dev.json"
            print(dev_data_file)
            dev_dataset = MyDataset(dev_data_file, params=dataset_params)
            dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        if args.do_test:
            ## rst
            # test_data_file = "data/dataset/eng.rst.gum/eng.rst.gum_test.json"
            # test_data_file = "data/dataset/eng.rst.rstdt/eng.rst.rstdt_test.json"
            # test_data_file = "data/dataset/eus.rst.ert/eus.rst.ert_test.json"
            # test_data_file = "data/dataset/fas.rst.prstc/fas.rst.prstc_test.json"
            # test_data_file = "data/dataset/nld.rst.nldt/nld.rst.nldt_test.json"
            # test_data_file = "data/dataset/por.rst.cstn/por.rst.cstn_test.json"
            # test_data_file = "data/dataset/rus.rst.rrt/rus.rst.rrt_test.json"
            # test_data_file = "data/dataset/spa.rst.rststb/spa.rst.rststb_test.json"
            # test_data_file = "data/dataset/spa.rst.sctb/spa.rst.sctb_test.json"
            # test_data_file = "data/dataset/zho.rst.sctb/zho.rst.sctb_test.json"
            ## dep
            # test_data_file = "data/dataset/eng.dep.scidtb/eng.dep.scidtb_test.json"
            ## pdtb
            # test_data_file = "data/dataset/tur.pdtb.tdb/tur.pdtb.tdb_test.json"
            # test_data_file = "data/dataset/tha.pdtb.tdtb/tha.pdtb.tdtb_test.json"
            # test_data_file = "data/dataset/eng.pdtb.pdtb/eng.pdtb.pdtb_test.json"
            ## sdrt
            test_data_file = "data/dataset/eng.sdrt.stac/eng.sdrt.stac_test.json"
            test_dataset = MyDataset(test_data_file, params=dataset_params)
            test_dataloader = get_dataloader(test_dataset, args, mode="test")

        for epoch in range(3, args.num_train_epochs+1):
            checkpoint_file = temp_file.format(str(epoch))
            print(" Epoch: {}".format(str(epoch)))
            print(checkpoint_file)
            args.output_dir = os.path.dirname(checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.do_dev:
                score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
                print(" Dev: Epoch=%d, Acc=%.4f, F1=%.4f\n" % (epoch, score_dict["acc_score"], score_dict["f1_score"]))
            if args.do_test:
                score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
                print(" Test: Epoch=%d, Acc=%.4f, F1=%.4f\n" % (epoch, score_dict["acc_score"], score_dict["f1_score"]))
            print()

if __name__ == "__main__":
    main()

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import pickle
import time

import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_example(filename):
    """Read examples from filename."""
    example = ""
    with open(filename, 'r', encoding='utf-8') as f1:
        example = f1.readlines()[0].strip()
    return example


def convert_example_to_feature(example, tokenizer):
    source_tokens = tokenizer.tokenize(example)[:512 - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = 512 - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length
    return source_ids, source_mask


def main():
    parser = argparse.ArgumentParser()

    ## parameters
    parser.add_argument("--model_type", default="codet5", type=str,
                        help="Type of pre-trained model: e.g. roberta,codebert,codet5")
    parser.add_argument("--model_name", default="Salesforce/codet5-base", type=str,
                        help="Name to pre-trained model: e.g. roberta-base,Salesforce/codet5-base")
    parser.add_argument("--model_dir", default="./models/model_for_application/", type=str,
                        help="Directory for models.")
    parser.add_argument("--langs", default="C#,C++,Java,JavaScript,Python", type=str,
                        help="The supported langs for translation.")
    parser.add_argument("--input_filename", default="./input.txt", type=str,
                        help="Path of input file.")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    args = parser.parse_args()
    langs = args.langs.split(',')
    logger.info(" %s = %s " % ("langs", " ".join(langs)))
    models_group = dict()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    for src_lang in langs:
        models_dict = dict()
        for tgt_lang in langs:
            if src_lang == tgt_lang:
                continue
            model_path = args.model_dir + src_lang + "2" + tgt_lang + "/checkpoint-last/pytorch_model.bin"
            logger.info("reload model from {}".format(model_path))
            model = model_class.from_pretrained(args.model_name)
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
            models_dict[tgt_lang] = model
        models_group[src_lang] = models_dict

    # main loop
    input_str = input()
    while not (input_str == "quit"):
        logger.info("%s = %s" % ("command is:", input_str))
        input_langs = input_str.split("to")
        if len(input_langs) != 2:
            input_str = input()
            continue
        lang0 = input_langs[0].strip()
        lang1 = input_langs[1].strip()
        if lang0 == lang1:
            input_str = input()
            continue
        if not (lang0 in langs and lang1 in langs):
            input_str = input()
            continue
        logger.info("%s to %s" % (lang0, lang1))
        logger.info("Read examples and get features")
        eval_example = read_example(args.input_filename)
        t1 = time.time()
        source_ids, source_mask = convert_example_to_feature(eval_example, tokenizer)
        all_source_ids = torch.tensor([source_ids], dtype=torch.long).to(device)
        all_source_mask = torch.tensor([source_mask], dtype=torch.long).to(device)
        logger.info("Begin generate")
        with torch.no_grad():
            preds = models_group[lang0][lang1].generate(all_source_ids,
                                                        attention_mask=all_source_mask,
                                                        use_cache=True,
                                                        num_beams=5,
                                                        early_stopping=True,
                                                        max_length=512)
            top_preds = list(preds.cpu().numpy())
        p = [tokenizer.decode(id_, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id_ in top_preds]
        t2 = time.time()
        logger.info("Cost time: %d" % float((t2-t1)*1000))
        logger.info("%s%s " % ("Output is:\n", "".join(p)))
        input_str = input()


if __name__ == "__main__":
    main()

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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")
    args = parser.parse_args()
    assert len(args.test_filename.split(',')) == 2
    print(args.test_filename)
    src_filename = args.test_filename.split(',')[0]
    trg_filename = args.test_filename.split(',')[1]
    dev_bleu = round(_bleu(src_filename, trg_filename), 2)
    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    # logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
    # logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()

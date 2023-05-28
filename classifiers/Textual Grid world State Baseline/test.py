from dataclasses import dataclass
import os
import datetime
from pathlib import Path
import shutil
import pickle
import numpy as np
import random
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import (
    AutoModelForSequenceClassification,
)
from tqdm.auto import tqdm

from preprocess.dataset_preprocess import get_tensor_dataset
from preprocess.tools.block_tools import (
    transform_block,
    count_block_colors,
    create_context_colour_count,
    get_color_counter_by_level,
    create_context_colour_by_height_level,
)

model_name = "derbertav3_base"
model_hug = "microsoft/deberta-v3-base"
max_seq_length = 320


class DEBERTAColourLevelContextClassifier:
    def __init__(self):
        print(torch.cuda.is_available())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_seq_length = 320

        epochs = 7
        use_epoch = 2
        lr = 1e-5
        model_hug = "microsoft/deberta-v3-base"
        model_name = "derbertav3_base_oversampling"  # "derbertav3_base"

        path = Path(__file__).parent / f"../../nb/saved_tokenizer_{model_hug}"
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.models = []
        for fold in range(5):
            pathzip = (
                Path(__file__).parent
                / f"../../nb/saved_model/{model_name}_{use_epoch}/{epochs}e_{lr}lr_f{fold}.zip"
            )
            path = (
                Path(__file__).parent
                / f"../../nb/saved_model/{model_name}_{use_epoch}/"
            )
            # shutil.unpack_archive(pathzip, path)
            import zipfile

            with zipfile.ZipFile(pathzip, "r") as zip_ref:
                zip_ref.extractall(path)
            path = (
                Path(__file__).parent
                / f"../../nb/saved_model/{model_name}_{use_epoch}/{epochs}e_{lr}lr_f{fold}/"
            )

            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def raise_aicrowd_error(self, msg):
        """Will be used by the evaluator to provide logs, DO NOT CHANGE"""
        raise NameError(msg)

    def _preprocess_context(self, gridworld_state):

        # state['avatarInfo']['pos'],state['avatarInfo']['look'],state

        blocks = [
            transform_block(block)
            for block in gridworld_state["worldEndingState"]["blocks"]
        ]
        blocks_by_level = get_color_counter_by_level(blocks)
        context = create_context_colour_by_height_level(blocks_by_level)

        return context

    def clarification_required(self, instruction, gridworld_state):
        """
        Implements classifier for given instuction - whether a clarifying question is required or not
        Inputs:
            instruction - Single instruction string

            gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                              NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

        Outputs:
            0 or 1 - 0 if clarification is not required, 1 if clarification is required

        """

        # get the context for the Transformer from the gridwold state
        context = self._preprocess_context(gridworld_state)

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        encoded_dict = self.tokenizer.encode_plus(
            context,
            instruction,
            add_special_tokens=True,
            max_length=self.max_seq_length,  # `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'`
            # pad_to_max_length = True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():

            inputs = encoded_dict["input_ids"].to(self.device)
            attention_mask = encoded_dict["attention_mask"].to(self.device)
            # print(inputs.shape,attention_mask.shape)
            pred_avg = 0
            for i in range(len(self.models)):
                pred = self.models[i](inputs, attention_mask=attention_mask)
                pred_avg += (
                    torch.softmax(pred.logits, axis=1).detach().cpu().numpy()[:, 1]
                ) / len(self.models)

        # return (
        #     pred_avg / len(self.models)
        # ) > 3 / 5  # np.argmax(results.logits.cpu().numpy(), axis=1)

        return pred_avg > 0.67

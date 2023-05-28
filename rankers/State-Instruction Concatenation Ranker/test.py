import random
from dataclasses import dataclass

from datasets import Dataset
from typing import Union, List, Tuple, Dict
import numpy as np
import pandas as pd

import torch
import torch.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from models.rankers.bert.src.common import SPECIAL_TOKENS_MAP
from models.data.state2desc import state2description, state2desc


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def CrossEncoderRankerFactory(model_path=None,
                              data_type=None,
                              use_no_block=False,
                              only_ins=False,
                              sep_token_type="word",
                              two_stage=False,
                              norm_type=None):

    class CrossEncoderRanker:
        """
        @data_type: None / gzb / gqy
        @sep_token_type: None / word / token
        @norm_type: None / Softmax / Sigmoid
        """

        def __init__(self,
                     model_path=model_path,
                     data_type=data_type,
                     use_no_block=use_no_block,
                     only_ins=only_ins,
                     sep_token_type=sep_token_type,
                     two_stage=two_stage) -> None:
            print("=" * 30, locals())  # print args
            self.model_path = model_path
            self.data_type = data_type
            self.use_no_block = use_no_block
            self.only_ins = only_ins
            self.sep_token_type = sep_token_type
            self.two_stage = two_stage
            self.norm_type = norm_type

            self.max_len = 512
            self.batch_size = 42
            self.device = "cuda"
            self._prepare_model_tokenizer(self.model_path)

        def raise_aicrowd_error(self, msg):
            """Will be used by the evaluator to provide logs, DO NOT CHANGE"""
            raise NameError(msg)

        def _prepare_model_tokenizer(self, model_path):
            if isinstance(self.model_path, str):
                self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                               use_fast=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path)
                self.model.eval()
                self.model.to(self.device)
            else:
                self.tokenizer = []
                self.model = []
                for p in model_path:
                    self.tokenizer.append(
                        AutoTokenizer.from_pretrained(p, use_fast=False))
                    self.model.append(
                        AutoModelForSequenceClassification.from_pretrained(p))
                # for m in self.model:
                #     m.eval()
                #     m.to(self.device)

        def _prepare_dataloader(self, tokenizer, input_tokens, question_bank):
            if self.sep_token_type == "token":
                input_tokens = [
                    f"{first}{second}"
                    for first, second in zip(input_tokens, question_bank)
                ]
                input_features = tokenizer(
                    input_tokens,
                    truncation=True,
                    max_length=self.max_len,
                    padding=False,
                )
            else:
                input_features = tokenizer(
                    input_tokens,
                    question_bank,
                    truncation="only_second",
                    max_length=self.max_len,
                    padding=False,
                )
            dataset = Dataset.from_dict(input_features)
            dataloader = DataLoader(dataset,
                                    collate_fn=GroupCollator(tokenizer),
                                    batch_size=self.batch_size,
                                    shuffle=False)
            return dataloader

        def _rank_questions(self, model, dataloader):
            total_scores = []
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                for inputs in dataloader:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with autocast():
                        outputs = model(**inputs)
                    logits = outputs.logits  # (batch_size, 1)
                    scores = logits.squeeze(-1)
                    if self.norm_type == "Sigmoid":
                        scores = torch.sigmoid(scores)
                    total_scores.extend(scores.detach().tolist())
            if self.norm_type == "Softmax":
                total_scores = softmax(np.array(total_scores))
            return total_scores

        def rank_questions_two_stage(self,
                                     input_tokens,
                                     question_bank,
                                     topk=30):
            down_ids = None
            for tokenizer, model in zip(self.tokenizer, self.model):
                torch.cuda.empty_cache()
                dataloader = self._prepare_dataloader(tokenizer, input_tokens,
                                                      question_bank)
                curr_scores = self._rank_questions(model, dataloader)
                ids = np.argsort(curr_scores)[::-1]
                if down_ids is None:
                    top_ids = ids[:topk]
                    down_ids = ids[topk:]
                    question_bank = [question_bank[i] for i in top_ids]
                    input_tokens = [input_tokens[i] for i in top_ids]
                else:
                    top_ids = [top_ids[i] for i in ids]
            top_ids = list(top_ids) + list(down_ids)
            return top_ids

        def rank_questions(self, instruction, gridworld_state, question_bank):
            """
            Implements the ranking function for a given instruction
            Inputs:
                instruction - Single instruction string, may or may not need any clarifying question
                            The evaluator may pass questions that don't need clarification,
                            But only questions requiring clarifying questions will be scored

                gridworld_state - Internal state from the iglu-gridworld simulator corresponding to the instuction
                                NOTE: The state will only contain the "avatarInfo" and "worldEndingState"

                question_bank - List of clarifying questions to rank

            Outputs:
                ranks - A sorted list of questions from the question bank
                        Such that the first index corresponds to the best ranked question

            """
            if self.sep_token_type is None:
                prefix_map = {"state": "", "instruction": "", "question": ""}
            else:
                prefix_map = SPECIAL_TOKENS_MAP[self.sep_token_type]

            if self.data_type is None:
                input_tokens = [
                    prefix_map["instruction"].lstrip() + instruction
                    for _ in question_bank
                ]
            else:
                if self.data_type == "gqy":
                    description = state2description(gridworld_state)
                elif self.data_type == "gzb":
                    description = state2desc(gridworld_state,
                                             instruction,
                                             in_instruction_only=self.only_ins,
                                             use_no_block=self.no_block)
                else:
                    raise NotImplementedError
                input_tokens = [
                    f"{prefix_map['state']}{description}{prefix_map['instruction']}{instruction}"
                    for _ in question_bank
                ]

            # sort by length, import speed by 8%
            question_bank.sort(key=lambda x: len(x))

            processed_question_bank = [
                f"{prefix_map['question']}{q}" for q in question_bank
            ]

            if isinstance(self.model_path, str):
                dataloader = self._prepare_dataloader(self.tokenizer,
                                                      input_tokens,
                                                      processed_question_bank)
                total_scores = self._rank_questions(self.model, dataloader)
            elif not self.two_stage:
                total_scores = None
                for tokenizer, model in zip(self.tokenizer, self.model):
                    torch.cuda.empty_cache()
                    # maybe different tokenizers
                    dataloader = self._prepare_dataloader(
                        tokenizer, input_tokens, processed_question_bank)
                    curr_scores = self._rank_questions(model, dataloader)
                    total_scores = curr_scores if total_scores is None else np.array(
                        total_scores) + np.array(curr_scores)
            else:
                ids = self.rank_questions_two_stage(input_tokens,
                                                    processed_question_bank)
                ranked_question_list = [question_bank[i] for i in ids]
                return ranked_question_list

            ids = np.argsort(total_scores)[::-1]
            ranked_question_list = [question_bank[i] for i in ids]

            return ranked_question_list
    return CrossEncoderRanker


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self,
                 dataset: Dataset,
                 tokenizer: PreTrainedTokenizer,
                 max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


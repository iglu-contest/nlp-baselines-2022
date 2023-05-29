from rank_bm25 import BM25Okapi, BM25Plus, BM25L
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from scipy.stats import rankdata
import os
from pathlib import Path


import torch
from transformers import (
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm.auto import tqdm

from preprocess.dataset_preprocess import get_tensor_dataset
from preprocess.tools.block_tools import (
    transform_block,
    get_color_counter_by_level,
    create_context_colour_by_height_level,
)

# from .collators import DataCollatorForMultipleChoice

nltk.data.path = [os.path.join(os.getcwd(), "models/rankers/nltk_data")]


def stem_tokenize(text, remove_stopwords=True):
    stemmer = PorterStemmer()
    tokens = [
        word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
    ]
    tokens = [
        word for word in tokens if word not in nltk.corpus.stopwords.words("english")
    ]
    return [stemmer.stem(word) for word in tokens]


class BM25DebertaColorizedRankerExtended:
    # Ther ewill be a problem when the classifier class does not get deleted
    # the models in device will not be deallocated. A way to solve this is to make the models globals to the class
    # then in the instantiation of this class replace the models array by None and call torch.cuda_empty_cache()
    def __init__(self):
        print(torch.cuda.is_available())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_seq_length = 320
        self.padding = True
        self.max_length = None
        self.pad_to_multiple_of = None
        self.qs_per_batch = 150
        self.num_pos_q = 300
        num_folds = 5
        epochs = 5
        epoch = 4
        lr = 7e-5
        epoch_folder = "best_from_each"  # "e_4"  # best_from_each

        model_hug = "microsoft/deberta-v3-base"
        model_name = "deberta_cv_full_recall_25_25_env_on_full_db120_augmentcolorper_wd5e-3_AWP_continued_awp2_3_7e-05lr_5"
        # _continued_awp2_3
        # "deberta_cv_full_recall_25_25_env_continued"  # "deberta_cv_continued"  # "deberta_cv_full_recall_25_25_env"  # "derbertav3_base"

        path = Path(__file__).parent / f"../../nb/saved_tokenizer_{model_hug}"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # /home/felipe/minerl2022/iglu/nlp/repo/iglu-2022-clariq-nlp-starter-kit/nb/saved_model/ranker_deberta_cv_full_recall_25_25_env_7e-05lr_5
        self.models = []
        for fold in range(num_folds):
            pathzip = (
                Path(__file__).parent
                / f"../../nb/saved_model/ranker_{model_name}/{epoch_folder}/f{fold}.zip"  # e_{epoch}/f{fold}.zip
            )
            path = (
                Path(__file__).parent
                / f"../../nb/saved_model/ranker_{model_name}/{epoch_folder}/"  # e_{epoch}/
            )
            # # shutil.unpack_archive(pathzip, path)
            # print("model_unzipping")
            import zipfile

            with zipfile.ZipFile(pathzip, "r") as zip_ref:
                zip_ref.extractall(path)
            path = (
                Path(__file__).parent
                / f"../../nb/saved_model/ranker_{model_name}/{epoch_folder}/f{fold}"  # best_from_each/f{fold}"  # e_{epoch}/f{fold}"
            )
            #  print("initializing model")
            model = AutoModelForMultipleChoice.from_pretrained(path)
            #  print("model to device")
            model.to(self.device)
            # print("model eval")
            model.eval()
            self.models.append(model)

    # print("all models loaded")

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

    def _instruction_question_preprocess(self, instruction, question):

        num_choices = len(question)

        first_sentences = [
            instruction
        ] * num_choices  # [[context] * len(num_pos_questions) for context in examples["instruction"]]

        second_sentences = question

        # first_sentences = sum(first_sentences, [])
        # second_sentences = sum(second_sentences, [])

        # should see if we should hand fix max_length and if to set the output as a pytorch tensor directly
        tokenized_examples = self.tokenizer(
            first_sentences, second_sentences, truncation=True, padding=True
        )

        return {
            k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
            for k, v in tokenized_examples.items()
        }

    def _get_batch_pred(self, context, instruction, ranked_question_head):
        # print("inside batch pred")
        choices = self._instruction_question_preprocess(
            context + self.tokenizer.sep_token + instruction, ranked_question_head
        )

        choices = [choices]
        batch_size = 1
        num_choices = len(ranked_question_head)
        # maybe change the collator so that it work with an is_inference flag
        ################collator########################################

        # @TODO we added a v[0][i] check why there is this unflattended level here that was not there before
        flattened_features = [
            [{k: v[0][i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in choices
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # print(batch)
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        #  print("batch to device starts")
        ####################################collator####################
        with torch.no_grad():

            inputs = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            # print(inputs.shape,attention_mask.shape)
            pred_avg = 0
            for i in range(len(self.models)):
                pred_avg += self.models[i](
                    inputs, attention_mask=attention_mask
                ).logits.detach().cpu().numpy() / len(self.models)
        return pred_avg

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
        # get the context for the Transformer from the gridwold state
        # print("preprocessing context")
        context = self._preprocess_context(gridworld_state)

        tokenized_questions = [stem_tokenize(q) for q in question_bank]
        token_question_map = {
            " ".join(tq): q for q, tq in zip(question_bank, tokenized_questions)
        }
        bm25 = BM25Plus(tokenized_questions)
        # bm25 = BM25Plus(tokenized_questions)
        tokenized_instruction = stem_tokenize(instruction, True)
        bm25_ranked_tokenized_questions = bm25.get_top_n(
            tokenized_instruction, tokenized_questions, n=len(tokenized_questions)
        )
        ranked_joined_sentences = [
            " ".join(tq) for tq in bm25_ranked_tokenized_questions
        ]
        ranked_question_list = [
            token_question_map[sent] for sent in ranked_joined_sentences
        ]

        if len(ranked_question_list) >= self.num_pos_q:
            ranked_question_tail = ranked_question_list[self.num_pos_q :]

            ranked_question_head = ranked_question_list[: self.num_pos_q]
            num_net_q = self.num_pos_q
        else:
            ranked_question_tail = []
            ranked_question_head = ranked_question_list
            num_net_q = len(ranked_question_head)

        preds = []
        # print("looping over qs per batch")
        for i in range(((num_net_q - 1) // self.qs_per_batch) + 1):
            questions_to_rank = ranked_question_head[
                self.qs_per_batch * i : min(self.qs_per_batch * (i + 1), num_net_q)
            ]
            preds.append(self._get_batch_pred(context, instruction, questions_to_rank))
        pred_avg = np.concatenate(preds, axis=1)

        # print("resorting")
        # resorting
        new_order = (num_net_q - rankdata(pred_avg)).astype(int)

        new_ranked_question_list = ranked_question_head.copy()

        for idx, q in zip(new_order, ranked_question_head):
            new_ranked_question_list[idx] = q
        # print("rearranged all qs")
        return new_ranked_question_list + ranked_question_tail

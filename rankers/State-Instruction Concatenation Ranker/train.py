# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import AutoConfig, AutoTokenizer
import transformers
from models.rankers.bert.src.arguments import (
    ModelArguments,
    DataArguments,
    RerankerTrainingArguments as TrainingArguments,
)
from models.rankers.bert.src.common import SPECIAL_TOKENS_MAP
from models.rankers.bert.src.data import ClsDataset, GroupedTrainDataset, PredictionDataset, GroupCollator
from models.rankers.bert.src.trainer import EvalEpochIntervalCallback, RerankerTrainer, RerankerDCTrainer
from models.rankers.bert.src.models.cross_encoder import Reranker, RerankerCls, RerankerDC, RerankerMultiTask
import sys
import logging
import os

root_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(root_dir))


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    # if training_args.distance_cache:
    #     _model_class = RerankerDC
    # elif training_args.only_cls:
    #     _model_class = RerankerCls
    # elif training_args.loss_weight != 1.0:
    _model_class = RerankerMultiTask
    config.problem_type = "multi_label_classification"
    # else:
    #     _model_class = Reranker

    model = _model_class.from_pretrained(
        model_args,
        data_args,
        training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=model_args.load_strict,
    )
    if data_args.sep_token_type == "token":
        tokenizer.add_tokens(
            list(SPECIAL_TOKENS_MAP[data_args.sep_token_type].values()))
        model.hf_model.resize_token_embeddings(len(tokenizer))

    _data_class = ClsDataset if training_args.only_cls else GroupedTrainDataset

    # Get datasets
    if training_args.do_train:
        train_dataset = _data_class(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args, is_train=True)
        valid_dataset = _data_class(
            data_args, data_args.dev_path, tokenizer=tokenizer, train_args=training_args, is_train=False)
        logger.info("Show training examples ...")
        logger.info(train_dataset[0])
        if training_args.only_cls:
            logger.info(tokenizer.batch_decode(
                [train_dataset[0]["input_ids"]], skip_special_tokens=True))
        else:
            logger.info(tokenizer.batch_decode(
                [train_dataset[0][0]["input_ids"]], skip_special_tokens=True))
        logger.info("Show valid examples ...")
        logger.info(valid_dataset[0])
        if training_args.only_cls:
            logger.info(tokenizer.batch_decode(
                [valid_dataset[0]["input_ids"]], skip_special_tokens=True))
        else:
            logger.info(tokenizer.batch_decode(
                [valid_dataset[0][0]["input_ids"]], skip_special_tokens=True))
    else:
        train_dataset = None
        valid_dataset = _data_class(
            data_args, data_args.dev_path, tokenizer=tokenizer, train_args=training_args)

    # Initialize our Trainer
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=GroupCollator(tokenizer),
        callbacks=[EvalEpochIntervalCallback],
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            with open(os.path.join(training_args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                f.write(str(trainer.state.best_metric))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

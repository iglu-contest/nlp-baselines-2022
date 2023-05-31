import os
import sys
import json
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from transformers import get_scheduler
from tensorboardX import SummaryWriter
from model import Builder
from dataset import IGLUDatasetClassifier, IGLUDatasetRanker
from utils import seed_torch, count_parameters, info_nce_loss, get_lr, eval_result, compute_mmr, mask_logits, collate_fn


def main(args, config):
    if not os.path.exists(args.saved_models_path):
        os.mkdir(args.saved_models_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] [%(levelname)s] %(message)s", datefmt="%H:%M:%S %a %b %d %Y")

    sHandler = logging.StreamHandler()
    sHandler.setLevel(logging.INFO)
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    fHandler = logging.FileHandler(os.path.join(args.saved_models_path, 'output.log'), mode='w')
    fHandler.setLevel(logging.INFO)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    writer = SummaryWriter(log_dir=args.saved_models_path)
    # f_writer = open(os.path.join(args.saved_models_path, 'output.txt'), 'w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using {}.'.format(device))
    num_gpus = torch.cuda.device_count()
    logger.info('Using {} GPUs'.format(num_gpus))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.task_name == "c":
        if args.use_annotated_dataset:
            iglu_annotated_dataset = IGLUDatasetClassifier(args.data_path, tokenizer, max_length=args.max_length, data_file_name="./public_data/iglu_nlptask_classifier_annotated_preprocessed_data_ml200.pik")
        iglu_dataset = IGLUDatasetClassifier(args.data_path, tokenizer, max_length=args.max_length)
        train_size, valid_size = int(args.train_size * len(iglu_dataset)), len(iglu_dataset) - int(args.train_size * len(iglu_dataset))
        train_dataset, valid_dataset = random_split(iglu_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
        if args.use_annotated_dataset:
            train_dataset = ConcatDataset([train_dataset, iglu_annotated_dataset])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    elif args.task_name == "r":
        iglu_dataset = IGLUDatasetRanker(args.data_path, tokenizer, max_length=args.max_length)
        full_corpus = iglu_dataset.full_corpus.to(device)
        train_size, valid_size = int(args.train_size * len(iglu_dataset)), len(iglu_dataset) - int(args.train_size * len(iglu_dataset))
        train_dataset, valid_dataset = random_split(iglu_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.seed))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
    else:
        sys.exit(0)
    logger.info('Train size: {}, Valid size: {}'.format(train_size, valid_size))

    model = Builder(config, freeze_encoder=args.freeze_encoder).to(device)
    if args.task_name == "r":
        docs_encoder = BertModel.from_pretrained("bert-base-uncased").to(device)
        params = list(model.parameters()) + list(docs_encoder.parameters())
        optimizer = AdamW(params, lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    num_training_steps = args.epochs * len(train_dataloader)
    if args.task_name == "c":
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)

    config['model_params'] = count_parameters(model)
    with open(os.path.join(args.saved_models_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    if args.task_name == 'c':
        max_f1 = 0
        train_max_f1 = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_data_num = 0
            train_preds = []
            train_labels = []
            for _, data in enumerate(tqdm(train_dataloader)):
                inputs, world_states, labels = data
                bs, _, _ = inputs['input_ids'].shape
                inputs = {k: v.view(bs, -1).to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                world_states = world_states.to(device)

                optimizer.zero_grad()
                logits, text_repr = model(inputs, world_states)
                train_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                train_labels.extend(labels.cpu().tolist())
                train_data_num += bs

                loss = criterion(logits, labels)
                train_loss += loss.sum().item()
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()

            train_precision, train_recall, train_f1, train_acc = eval_result(train_preds, train_labels)
            train_loss = train_loss / train_data_num
            if train_f1 > train_max_f1:
                logger.info("Saving trained model at epoch {} ...".format(epoch))
                torch.save(model.state_dict(), os.path.join(args.saved_models_path, "classifier_train_model.pt"))
                train_max_f1 = train_f1

            with torch.no_grad():
                model.eval()
                valid_loss = 0
                valid_data_num = 0
                valid_preds = []
                valid_labels = []
                for _, data in enumerate(tqdm(valid_dataloader)):
                    inputs, world_states, labels = data
                    bs, _, _ = inputs['input_ids'].shape
                    inputs = {k: v.view(bs, -1).to(device) for k, v in inputs.items()}
                    labels = labels.to(device)
                    world_states = world_states.to(device)

                    logits, text_repr = model(inputs, world_states)
                    valid_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                    valid_labels.extend(labels.cpu().tolist())
                    valid_data_num += bs

                    loss = criterion(logits, labels)
                    valid_loss += loss.sum().item()

                valid_precision, valid_recall, valid_f1, valid_acc = eval_result(valid_preds, valid_labels)
                valid_loss = valid_loss / valid_data_num
                if valid_f1 > max_f1:
                    logger.info("Saving model at epoch {} ...".format(epoch))
                    torch.save(model.state_dict(), os.path.join(args.saved_models_path, "classifier_model.pt"))
                    max_f1 = valid_f1

            current_lr = get_lr(optimizer)
            writer.add_scalars("F1", {"Train": train_f1, "Valid": valid_f1}, epoch)
            writer.add_scalars("Loss", {"Train": train_loss, "Valid": valid_loss}, epoch)
            logger.info('Train Epoch {} | Acc {}, P: {}, R: {}, F1: {}, Loss: {}'.format(epoch, train_acc, train_precision, train_recall, train_f1, train_loss))
            logger.info('Valid Epoch {} | Acc {}, P: {}, R: {}, F1: {}, Loss: {}'.format(epoch, valid_acc, valid_precision, valid_recall, valid_f1, valid_loss))
            logger.info('Current lr: {}\n'.format(current_lr))
    else:
        max_mrr = 0
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_data_num = 0
            for _, data in enumerate(tqdm(train_dataloader)):
                inputs, world_states, candidate_inputs, _, _ = collate_fn(data, device)
                # inputs: [bsz, max_length=100],
                # candidate_inputs: [bsz, max_length=100],
                # world_states: [bsz, 7, 11, 9, 11]
                bs, _ = inputs['input_ids'].shape

                optimizer.zero_grad()
                _, queries = model(inputs, world_states)  # [bsz, hidden_size=768]
                docs = docs_encoder(**candidate_inputs).pooler_output  # [bsz, hidden_size=768]
                logits = info_nce_loss(queries, docs, temperature=args.temperature)  # [bsz, bsz]
                train_data_num += bs

                loss = criterion(logits, torch.tensor(range(bs)).to(device))
                train_loss += loss.sum().item()
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()

            train_loss = train_loss / train_data_num

            with torch.no_grad():
                model.eval()
                valid_loss = 0
                valid_data_num = 0
                first = True
                valid_ground_truth_idxs = []
                for _, data in enumerate(tqdm(valid_dataloader)):
                    inputs, world_states, _, labels, temp_q_bank = collate_fn(data, device)
                    bs, _ = inputs['input_ids'].shape

                    _, queries = model(inputs, world_states)  # [bsz, hidden_size=768]
                    full_docs = docs_encoder(**full_corpus).pooler_output  # [num_docs, hidden_size=768]
                    logits = info_nce_loss(queries, full_docs, temperature=args.temperature)  # [bsz, num_docs]
                    logits = mask_logits(logits, temp_q_bank)
                    _, top_idxs = torch.topk(logits, k=20, dim=-1)
                    if first:
                        valid_top_idxs, first = top_idxs, False
                    else:
                        valid_top_idxs = torch.cat([valid_top_idxs, top_idxs], dim=0)
                    valid_ground_truth_idxs.extend(labels.tolist())
                    valid_data_num += bs

                    loss = criterion(logits, torch.tensor(range(bs)).to(device))
                    valid_loss += loss.sum().item()

                valid_mrr_score_dict = compute_mmr(valid_top_idxs, valid_ground_truth_idxs)
                valid_loss = valid_loss / valid_data_num
                if valid_mrr_score_dict["MRR@5"] > max_mrr:
                    logger.info("Saving model at epoch {} ...".format(epoch))
                    torch.save(model.state_dict(), os.path.join(args.saved_models_path, "classifier_model.pt"))
                    max_mrr = valid_mrr_score_dict["MRR@5"]

            current_lr = get_lr(optimizer)
            writer.add_scalars("Loss", {"Train": train_loss, "Valid": valid_loss}, epoch)
            logger.info('Train Epoch {} | Loss: {}'.format(epoch, train_loss))
            logger.info('Valid Epoch {} | Loss: {}, MRR@5: {}, MRR@10: {}, MRR@20: {}'
                        .format(epoch, valid_loss, valid_mrr_score_dict["MRR@5"], valid_mrr_score_dict["MRR@10"], valid_mrr_score_dict["MRR@20"]))
            logger.info('Current lr: {}\n'.format(current_lr))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--max_length', default=100, type=int)
    parser.add_argument('--config', type=str, default='models/config.json')
    parser.add_argument('--freeze_encoder', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--use_annotated_dataset', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, default='./public_data')
    parser.add_argument('--saved_models_path', type=str, default='output/o')
    parser.add_argument('--task_name', type=str, default='r', help='Please select c or r for classification or ranking.')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config["train_config"]['lr'] = args.lr
    config["train_config"]['seed'] = args.seed
    config["train_config"]['epochs'] = args.epochs
    config["train_config"]['train_size'] = args.train_size
    config["train_config"]['temperature'] = args.temperature
    config["train_config"]['batch_size'] = args.batch_size
    config["train_config"]['max_length'] = args.max_length
    config["train_config"]['freeze_encoder'] = args.freeze_encoder
    config["train_config"]['test'] = args.test

    seed_torch(args.seed)
    main(args, config)

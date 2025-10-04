#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning Multiple-Choice (SWAG style) with Accelerate — print & log train_loss/val_loss/val_acc
- 新增：每個 epoch 結束後，計算並列印 train_loss、val_loss、val_acc
- 新增：把歷史結果存成 <output_dir>/metrics.csv 與 <output_dir>/metrics.json
- 保留：best accuracy checkpoint 儲存至 <output_dir>/best
"""

import argparse, json, logging, math, os
from itertools import chain
from pathlib import Path

import datasets, evaluate, torch, transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, AutoModelForMultipleChoice, AutoTokenizer,
    DataCollatorForMultipleChoice, SchedulerType, default_data_collator, get_scheduler
)

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    p = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    # 原參數
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--validation_file", type=str, default=None)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--pad_to_max_length", action="store_true")
    p.add_argument("--model_name_or_path", type=str, required=False)
    p.add_argument("--config_name", type=str, default=None)
    p.add_argument("--tokenizer_name", type=str, default=None)
    p.add_argument("--use_slow_tokenizer", action="store_true")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                   choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    p.add_argument("--num_warmup_steps", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str)
    p.add_argument("--hub_token", type=str)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--checkpointing_steps", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--with_tracking", action="store_true")
    p.add_argument("--report_to", type=str, default="all")
    # best-model 儲存控制
    p.add_argument("--save_best", action="store_true")
    p.add_argument("--metric_for_best", type=str, default="accuracy")
    p.add_argument("--greater_is_better", action="store_true")
    p.add_argument("--save_best_each_epoch", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # output_dir
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 載入資料
    if args.dataset_name is not None:
        raw = load_dataset(args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code)
    else:
        from datasets import Dataset, DatasetDict
        def load_custom(train_file, valid_file):
            import json
            with open(train_file, "r", encoding="utf-8") as f: train_data = json.load(f)
            with open(valid_file, "r", encoding="utf-8") as f: valid_data = json.load(f)
            return DatasetDict({"train": Dataset.from_list(train_data), "validation": Dataset.from_list(valid_data)})
        raw = load_custom(args.train_file, args.validation_file)

    # 讀 context.json（如需改路徑請自行調整）
    with open("C:/Users/USER/mnt/data/context.json", "r", encoding="utf-8") as f:
        CONTEXTS = json.load(f)

    def reshape(example):
        para_ids = example["paragraphs"]
        endings = [CONTEXTS[i] for i in para_ids]
        example["sent1"]   = example["question"]
        example["sent2"]   = ""
        example["ending0"] = endings[0]
        example["ending1"] = endings[1]
        example["ending2"] = endings[2]
        example["ending3"] = endings[3]
        example["label"]   = para_ids.index(example["relevant"])
        return example

    for split in list(raw.keys()):
        raw[split] = raw[split].map(reshape)

    if args.debug:
        for split in raw:
            raw[split] = raw[split].select(range(100))

    column_names = raw["train"].column_names if "train" in raw else raw["validation"].column_names
    ending_names = ["ending0","ending1","ending2","ending3"]
    context_name = "sent1"
    question_header_name = "sent2"
    label_column_name = "label"

    # 模型與 tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("Instantiating new config from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer,
                                                  trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer,
                                                  trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError("Tokenizer must come from a pretrained name.")

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, config=config,
                                                           trust_remote_code=args.trust_remote_code)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config, trust_remote_code=args.trust_remote_code)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # 前處理
    padding = "max_length" if args.pad_to_max_length else False
    def preprocess_function(examples):
        first_sentences = [[ctx]*4 for ctx in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [[f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
        labels = examples[label_column_name]
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))
        enc = tokenizer(first_sentences, second_sentences, max_length=args.max_seq_length,
                        padding=padding, truncation=True)
        enc = {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in enc.items()}
        enc["labels"] = labels
        return enc

    with Accelerator().main_process_first():
        processed = raw.map(preprocess_function, batched=True, remove_columns=column_names)
    train_dataset = processed["train"]
    eval_dataset  = processed["validation"]

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        pad_to_multiple_of = 16 if accelerator.mixed_precision == "fp8" else (8 if accelerator.mixed_precision != "no" else None)
        data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator,
                              batch_size=args.per_device_train_batch_size)
    eval_loader  = DataLoader(eval_dataset, collate_fn=data_collator,
                              batch_size=args.per_device_eval_batch_size)

    # Optimizer / Scheduler
    no_decay = ["bias","LayerNorm.weight"]
    param_groups = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate)

    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    overrode = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(args.max_train_steps if overrode else args.max_train_steps * accelerator.num_processes),
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 追蹤 best
    best_metric_value = float("-inf") if args.greater_is_better else float("inf")
    best_dir = os.path.join(args.output_dir or ".", "best"); os.makedirs(best_dir, exist_ok=True)

    # CSV/JSON 紀錄
    metrics_rows = []
    csv_path = os.path.join(args.output_dir or ".", "metrics.csv")
    json_path = os.path.join(args.output_dir or ".", "metrics.json")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        # -------- Train --------
        model.train()
        running_loss = 0.0
        batches_in_epoch = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            running_loss += loss.detach().float()
            batches_in_epoch += 1

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            if completed_steps >= args.max_train_steps:
                break

        avg_train_loss = (running_loss / max(1, batches_in_epoch)).item()

        # -------- Eval --------
        model.eval()
        metric = evaluate.load("accuracy")
        eval_loss = 0.0
        n_batches = 0
        n_examples = 0
        correct = 0
        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(**batch)
                eval_loss += outputs.loss.detach().float()
                n_batches += 1
                preds = outputs.logits.argmax(dim=-1)
                preds, refs = accelerator.gather_for_metrics((preds, batch["labels"]))
                metric.add_batch(predictions=preds, references=refs)
                correct += (preds == refs).sum().item()
                n_examples += refs.size(0)
        avg_val_loss = (eval_loss / max(1, n_batches)).item()
        eval_metric = metric.compute()  # {'accuracy': ...}
        val_acc = float(eval_metric.get("accuracy", correct / max(1, n_examples)))

        accelerator.print(
            f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # 記錄到記憶體 → 末端存檔
        metrics_rows.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": val_acc})

        # 儲存 best
        improved = (val_acc > best_metric_value) if args.greater_is_better else (val_acc < best_metric_value)
        if args.save_best and improved:
            best_metric_value = val_acc
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(best_dir, is_main_process=True, save_function=accelerator.save)
                tokenizer.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, "accuracy": val_acc}, f, ensure_ascii=False, indent=2)
                accelerator.print(f"✅ Saved BEST model to: {best_dir}  (accuracy={val_acc:.6f})")

        if args.checkpointing_steps == "epoch":
            out_dir = os.path.join(args.output_dir or ".", f"epoch_{epoch}")
            accelerator.save_state(out_dir)

    # ------ 結束：存 CSV/JSON 與最後權重 ------
    if accelerator.is_main_process and args.output_dir is not None:
        # CSV
        try:
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc"])
                w.writeheader(); w.writerows(metrics_rows)
        except Exception:
            pass
        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_rows, f, ensure_ascii=False, indent=2)

        # 存最後一份（非必然最佳）
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir, is_main_process=True, save_function=accelerator.save)
        tokenizer.save_pretrained(args.output_dir)

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()

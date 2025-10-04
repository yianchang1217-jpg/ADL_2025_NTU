# -*- coding: utf-8 -*-
# run_swag_no_trainer_from_scratch.py
# Train a 4-choice paragraph selection model FROM SCRATCH (random init).
# Sample format per item:
# {
#   "id": str,                        # (僅供你日後對照，訓練時不吃，不要進 collator)
#   "question": str,
#   "paragraphs": [pid0, pid1, pid2, pid3],
#   "relevant": (0..3) OR (pid),      # 可給索引或直接給 pid；程式會自動對應
#   ... (其他欄位會被忽略)
# }
#
# context.json 可以是：
# - list: ["para0", "para1", ...]，此時 pid 應為整數索引
# - dict:  { pid: "text", ... }，pid 可為字串或整數鍵
#
# 產出：最佳 checkpoint 會存到 --output_dir/best

import os
import math
import json
import time
import argparse
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from accelerate import Accelerator

from transformers import (
    BertConfig,
    AutoTokenizer,
    AutoModelForMultipleChoice,
    get_scheduler,
    set_seed,
    DataCollatorForMultipleChoice,
)
from torch.optim import AdamW


# ---------------- I/O ----------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paragraphs_to_texts(par_ids: List[Any], contexts: Any) -> List[str]:
    """
    contexts 支援 list 或 dict：
      - list: 以 int(pid) 當索引
      - dict: 依序嘗試 pid、str(pid)、int(pid)
    """
    texts: List[str] = []

    if isinstance(contexts, list):
        n = len(contexts)
        for pid in par_ids:
            try:
                idx = int(pid)
            except (ValueError, TypeError):
                raise KeyError(f"paragraph pid '{pid}' 不是有效的整數索引（list 長度={n}）")
            if not (0 <= idx < n):
                raise KeyError(f"paragraph pid '{pid}' 超出 list 範圍 (0..{n-1})")
            texts.append(contexts[idx])
        return texts

    if isinstance(contexts, dict):
        for pid in par_ids:
            if pid in contexts:
                texts.append(contexts[pid]); continue
            spid = str(pid)
            if spid in contexts:
                texts.append(contexts[spid]); continue
            try:
                ipid = int(pid)
                if ipid in contexts:
                    texts.append(contexts[ipid]); continue
            except (ValueError, TypeError):
                pass
            raise KeyError(f"paragraph pid {pid} 在 context.json 中找不到（試過 {repr(pid)}, {repr(spid)} 以及 int(pid)）。")
        return texts

    raise TypeError(f"context.json 型別不支援: {type(contexts)}")


def resolve_relevant_index(relevant: Any, par_ids: List[Any]) -> int:
    """
    relevant 可以是 0..3 的索引，或是 par_ids 裡的一個 pid。
    先嘗試把 relevant 視為 pid 對應到 par_ids 的位置，失敗再當索引。
    """
    rel_str = str(relevant)
    for i, pid in enumerate(par_ids):
        if pid == relevant or str(pid) == rel_str:
            return i
    if isinstance(relevant, int) and 0 <= relevant < len(par_ids):
        return relevant
    raise ValueError(
        f"relevant={relevant} 無法在 paragraphs={par_ids} 中定位為 pid，且不屬於合法索引(0..{len(par_ids)-1})."
    )


# -------------- Feature building --------------

def build_multiple_choice_features(
    data: List[Dict[str, Any]],
    contexts: Any,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    """
    只輸出模型要的欄位，避免字串混進 collator：
      - input_ids: List[List[int]]  shape [num_choices, seq_len]
      - attention_mask: same
      - token_type_ids: optional
      - labels: int
    """
    records: List[Dict[str, Any]] = []

    for ex in data:
        question: str = ex["question"]
        par_ids: List[Any] = ex["paragraphs"]
        relevant: Any = ex["relevant"]

        texts = paragraphs_to_texts(par_ids, contexts)
        label_idx = resolve_relevant_index(relevant, par_ids)

        # 使用 fast tokenizer 的 __call__ 一次處理 truncation（padding 交給 collator）
        enc = tokenizer(
            [question] * len(texts),
            texts,
            truncation=True,
            max_length=max_seq_length,
        )

        feat: Dict[str, Any] = {
            "labels": int(label_idx),
            "input_ids": [],
            "attention_mask": [],
        }
        if "token_type_ids" in enc and enc["token_type_ids"] is not None:
            feat["token_type_ids"] = []

        for i in range(len(texts)):
            feat["input_ids"].append(enc["input_ids"][i])
            feat["attention_mask"].append(enc["attention_mask"][i])
            if "token_type_ids" in feat:
                feat["token_type_ids"].append(enc["token_type_ids"][i])

        # 注意：不把 'id' 放進去，避免 collator→tokenizer.pad 嘗試轉 tensor 失敗
        records.append(feat)

    return Dataset.from_list(records)


# -------------- Train / Eval --------------

def train_and_eval(
    args,
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer,
):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
        if args.gradient_accumulation_steps > 1 else 1
    )
    set_seed(args.seed)

    # 從零建立一個小型 BERT 結構（只借 tokenizer 的 vocab，沒有載入權重）
    assert args.hidden_size % args.num_attention_heads == 0, \
        "hidden_size 必須可被 num_attention_heads 整除"
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=(args.intermediate_size or (4 * args.hidden_size)),
        max_position_embeddings=args.max_position_embeddings,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        num_labels=4,
    )
    model = AutoModelForMultipleChoice.from_config(config)

    # 正確的 collator（會自己做 padding/truncation 對齊）
    data_collator = DataCollatorForMultipleChoice(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if accelerator.mixed_precision != "no" else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Optimizer / Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.98))

    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)))
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Accelerator 準備
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- Loop ----
    best_acc = -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    log_every = max(20, args.gradient_accumulation_steps)
    global_step = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if args.max_grad_norm and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().float()
            global_step += 1

            if step % log_every == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                accelerator.print(
                    f"Epoch {epoch} | step {step}/{len(train_loader)} | lr={current_lr:.6f} | loss={loss.item():.4f}"
                )

        # ----- Eval -----
        model.eval()
        eval_loss = 0.0
        n_batches = 0
        n_examples = 0
        correct = 0

        for batch in eval_loader:
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.detach().float()
                n_batches += 1
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                n_examples += batch["labels"].size(0)

        avg_train_loss = (total_loss / max(1, len(train_loader))).item()
        avg_eval_loss = (eval_loss / max(1, n_batches)).item()
        val_acc = correct / max(1, n_examples)

        accelerator.print(
            f"Epoch {epoch} finished in {time.time() - t0:.1f}s | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_eval_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            accelerator.print(f"New best val_acc = {best_acc:.4f} → saving to {args.output_dir}/best")
            if accelerator.is_main_process:
                save_dir = os.path.join(args.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

    accelerator.print(f"Training done. Best val_acc = {best_acc:.4f} (saved under {args.output_dir}/best)")


# -------------- CLI --------------

def parse_args():
    p = argparse.ArgumentParser(description="Train 4-choice paragraph selection FROM SCRATCH (random init).")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, required=True)
    p.add_argument("--context_file", type=str, required=True)

    # tokenizer 只借詞彙，不載權重
    p.add_argument("--tokenizer_name", type=str, default="bert-base-chinese")

    # model size（小型）
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_attention_heads", type=int, default=6)
    p.add_argument("--intermediate_size", type=int, default=None)
    p.add_argument("--max_position_embeddings", type=int, default=512)

    # optimization
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_warmup_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                   choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                            "constant", "constant_with_warmup"])

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--dataloader_num_workers", type=int, default=0)  # 先 0 好 debug，再視情況調高

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 讀資料
    train_data = load_json(args.train_file)
    valid_data = load_json(args.validation_file)
    contexts = load_json(args.context_file)

    # 只借 tokenizer 的 vocab
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # 建 feature（移除所有非模型輸入欄位，避免字串混入）
    train_ds = build_multiple_choice_features(train_data, contexts, tokenizer, args.max_seq_length)
    eval_ds  = build_multiple_choice_features(valid_data, contexts, tokenizer, args.max_seq_length)

    # 訓練
    train_and_eval(args, train_ds, eval_ds, tokenizer)


if __name__ == "__main__":
    main()

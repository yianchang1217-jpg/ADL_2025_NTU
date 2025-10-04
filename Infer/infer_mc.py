#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiple-Choice 推論腳本（SWAG/段落選擇）
- 輸入：test.json（每筆含 id, question, paragraphs[候選段落ID列表]）、context.json（id->段落文字）
- 載入：指定 model_dir（如訓練輸出的 best/ 或最後權重）
- 輸出：pred_map (dict) => {question_id: chosen_paragraph_id}
- 額外：印出統計，若 test.json 有 "relevant" 也會自動計算 accuracy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="模型與 tokenizer 的目錄（例如 output/lert_mc_best 或 lert_mc_scratch）")
    ap.add_argument("--test_json", type=str, required=True, help="測試檔 (JSON list)，每筆至少含 id, question, paragraphs")
    ap.add_argument("--context_json", type=str, required=True, help="段落內容的 JSON（id->text）")
    ap.add_argument("--out_json", type=str, required=True, help="輸出的預測 JSON 路徑")
    ap.add_argument("--batch_size", type=int, default=16, help="推論批次大小")
    ap.add_argument("--max_length", type=int, default=512, help="tokenize 的最大長度")
    ap.add_argument("--fp16", action="store_true", help="推論使用半精度（需 GPU）")
    ap.add_argument("--preview_rows", type=int, default=5, help="在終端列印前幾筆預測預覽")
    ap.add_argument("--device", type=str, default=None, help="指定裝置：cuda/cpu。預設自動偵測")
    return ap.parse_args()


def encode_example(tokenizer, question: str, paragraph_ids: List[int], contexts: Dict[str, str], max_len=512):
    """
    把 question 與 N 個候選段落組成 N 個 pair，回傳 MC 需要的 shape：[1, N, seq_len]
    """
    first = [question] * len(paragraph_ids)
    second = [contexts[str(pid)] if str(pid) in contexts else contexts[pid] for pid in paragraph_ids]
    enc = tokenizer(
        first, second,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return {k: v.view(len(paragraph_ids), -1).unsqueeze(0) for k, v in enc.items()}  # [1, N, L]


def main():
    args = parse_args()

    # ========= 裝置與精度 =========
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.fp16 and device.type == "cuda")

    # ========= 載入模型與 tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_dir).to(device)
    model.eval()

    # ========= 載入資料 =========
    with open(args.test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(args.context_json, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    # ========= 推論 =========
    pred_map: Dict[str, int] = {}
    total = len(test_data)
    batch_inputs = []
    batch_meta = []  # [(qid, para_ids)]

    def flush_batch():
        """把暫存的 batch_inputs 送進模型，寫回 pred_map。"""
        if not batch_inputs:
            return
        batch = {k: torch.cat([ex[k] for ex in batch_inputs], dim=0).to(device)
                 for k in batch_inputs[0].keys()}
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(**batch).logits  # [B, C]
            pred_idx = logits.argmax(dim=-1).tolist()
        for (qid, para_ids), idx in zip(batch_meta, pred_idx):
            pred_map[qid] = para_ids[idx]
        batch_inputs.clear()
        batch_meta.clear()

    # 允許不同題目的選項數相同即可一起 batch；若遇到不同 C，就先 flush。
    current_choices = None
    for item in test_data:
        qid = str(item["id"])
        question = item["question"]
        para_ids = item["paragraphs"]
        if not isinstance(para_ids, list) or len(para_ids) == 0:
            raise ValueError(f"[{qid}] 'paragraphs' 格式錯誤：{para_ids}")

        enc = encode_example(tokenizer, question, para_ids, contexts, max_len=args.max_length)

        C = len(para_ids)
        if current_choices is None:
            current_choices = C
        # 若本題選項數與目前 batch 選項數不同，先送出目前 batch
        if C != current_choices:
            flush_batch()
            current_choices = C

        batch_inputs.append(enc)
        batch_meta.append((qid, para_ids))

        if len(batch_inputs) == args.batch_size:
            flush_batch()

    # flush 剩餘
    flush_batch()

    # ========= 輸出與統計 =========
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred_map, f, ensure_ascii=False, indent=2)

    total_rows = len(test_data)
    pred_rows = len(pred_map)
    miss_rows = total_rows - pred_rows

    print(f"已輸出預測到: {out_path}")
    print(f"總測試筆數: {total_rows}")
    print(f"成功預測筆數: {pred_rows}")
    print(f"缺漏筆數: {miss_rows}")

    # 若 test.json 內含 'relevant' 欄位，順便計算 accuracy
    if total_rows > 0 and "relevant" in test_data[0]:
        correct = 0
        for ex in test_data:
            qid = str(ex["id"])
            gold = ex["relevant"]
            pred = pred_map.get(qid, None)
            if pred is not None and pred == gold:
                correct += 1
        acc = 100.0 * correct / total_rows
        print(f"Accuracy（含 'relevant' 的資料集）: {acc:.2f}%  ({correct}/{total_rows})")

    # 預覽前幾筆
    if args.preview_rows > 0:
        print("\n=== 預覽（前幾筆）===")
        shown = 0
        for ex in test_data:
            if shown >= args.preview_rows:
                break
            qid = str(ex["id"])
            para_ids = ex["paragraphs"]
            pred_pid = pred_map.get(qid, None)
            print(f"- id={qid}  預測段落={pred_pid}  候選={para_ids}")
            shown += 1


if __name__ == "__main__":
    main()

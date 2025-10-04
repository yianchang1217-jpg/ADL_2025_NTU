# -*- coding: utf-8 -*-
"""
build_and_infer.py
功能：
1) 將 test.json + context.json + mc_predictions.json 轉成 SQuAD v1.1 格式 test_squad.json
2) 對 SQuAD 做「批次」QA 推論，輸出 CSV（id,answer）

重點：
- 全程 Torch-only（不依賴 NumPy，避免 NumPy 2.x 相容性問題）
- 支援 --batch 批次推論；支援 --fp16（僅 GPU）
- 長文用滑窗（doc_stride），離線/本地模型可用 local_files_only

用法範例：
python build_and_infer.py ^
  --model_name_or_path .\output\lert_qa_large ^
  --test_json .\test.json ^
  --context_json .\context.json ^
  --mc_pred_json .\output\mc_predictions.json ^
  --out_squad_json .\output\test_squad.json ^
  --out_csv .\output\test_predictions.csv ^
  --max_seq_length 512 --doc_stride 128 --batch 32 --fp16
"""

import os
import csv
import json
import argparse
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# -----------------------------
# Part A: 轉換成 SQuAD v1.1
# -----------------------------
def build_squad_from_mc(
    test_json_path: str,
    context_json_path: str,
    mc_pred_json_path: str,
    out_squad_path: str,
    title: str = "test"
) -> Dict[str, Any]:
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(context_json_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    with open(mc_pred_json_path, "r", encoding="utf-8") as f:
        mc_pred = json.load(f)

    def get_context_by_pid(pid):
        try:
            return contexts[pid]
        except (KeyError, TypeError, IndexError):
            raise KeyError(
                f"無法用 pid={pid} 從 context_json 取得內容。"
                f"請確認 context.json 是否為 dict/list 且 pid 存在。"
            )

    squad = {"version": "v1.1", "data": []}
    n_ok, n_skip = 0, 0

    for item in test_data:
        qid = item["id"]
        question = item["question"]

        key = str(qid)
        if key not in mc_pred:
            n_skip += 1
            continue

        pid = mc_pred[key]
        context_text = get_context_by_pid(pid)

        squad["data"].append({
            "title": title,
            "paragraphs": [{
                "context": context_text,
                "qas": [{
                    "id": str(qid),
                    "question": question,
                    "answers": []
                }]
            }]
        })
        n_ok += 1

    out_dir = os.path.dirname(out_squad_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_squad_path, "w", encoding="utf-8") as f:
        json.dump(squad, f, ensure_ascii=False)

    print(f"[SQuAD] 轉換完成：{n_ok} 筆，跳過 {n_skip} 筆（mc 無對應 qid）。寫入：{out_squad_path}")
    return squad


# -----------------------------
# Part B: 批次 QA 推論
# -----------------------------
def build_features_from_squad(
    tokenizer,
    records: List[Tuple[str, str, str]],  # (qid, question, context)
    max_seq_length: int,
    doc_stride: int
):
    """
    將所有題目展開為 feature 清單（含 overflow 切片），
    回傳一個 dict：包含 input_ids, attention_mask, offset_mapping, example_id。
    """
    # 將 question/context 拆成列表給 tokenizer 做 batched encode
    questions = [q for _, q, _ in records]
    contexts  = [c for _, _, c in records]

    enc = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding="max_length",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    overflow = enc.pop("overflow_to_sample_mapping")
    # 建立 example_id 與 offset_mapping（只保留 context 區段）
    example_ids = []
    new_offsets = []
    ids = [qid for qid, _, _ in records]

    for i in range(len(overflow)):
        sample_idx = overflow[i]
        qid = ids[sample_idx]
        example_ids.append(qid)

        seq_ids = enc.sequence_ids(i)  # 0=question, 1=context, None=special
        offsets = enc["offset_mapping"][i]
        # 非 context 的位置設為 None，這樣後處理能快速過濾
        new_offsets.append([o if seq_ids[j] == 1 else None for j, o in enumerate(offsets)])

    features = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offset_mapping": new_offsets,
        "example_id": example_ids,
    }
    return features


def collate_fn(batch):
    """
    DataLoader 的整理函式。
    - 把 input_ids / attention_mask 轉 tensor
    - offset_mapping / example_id 保持 Python list 以便後處理
    """
    out = {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "offset_mapping": [b["offset_mapping"] for b in batch],
        "example_id": [b["example_id"] for b in batch],
    }
    return out


@torch.no_grad()
def infer_qa_on_squad_batched(
    model_name_or_path: str,
    test_squad_json_path: str,
    out_csv_path: str,
    tokenizer_name: str = None,
    max_seq_length: int = 512,
    doc_stride: int = 128,
    max_answer_len: int = 64,
    batch: int = 32,
    fp16: bool = False,
    local_files_only: bool = False,
) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_name = tokenizer_name or model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True, local_files_only=local_files_only)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, local_files_only=local_files_only).to(device)
    model.eval()

    # 讀 SQuAD，展平成 (qid, question, context) 列表
    with open(test_squad_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Tuple[str, str, str]] = []
    for art in data.get("data", []):
        for para in art.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                qid = str(qa["id"])
                question = qa["question"]
                records.append((qid, question, context))

    # 先一次性把所有題目展成 features（含 overflow 切片）
    feats = build_features_from_squad(tokenizer, records, max_seq_length, doc_stride)

    # 準備 DataLoader：以「feature」為單位批次推論
    feat_ds = [
        {
            "input_ids": feats["input_ids"][i],
            "attention_mask": feats["attention_mask"][i],
            "offset_mapping": feats["offset_mapping"][i],
            "example_id": feats["example_id"][i],
        }
        for i in range(len(feats["input_ids"]))
    ]
    dl = DataLoader(feat_ds, batch_size=batch, shuffle=False, collate_fn=collate_fn)

    # 先把所有 feature 的 logits 跑完、收在 CPU
    start_list, end_list = [], []
    pbar = tqdm(total=len(feat_ds), desc="Inference (features)", unit="feat")
    amp_ctx = torch.cuda.amp.autocast if (fp16 and device == "cuda") else torch.cpu.amp.autocast
    # torch.cpu.amp.autocast 是 no-op，這樣寫就不用分支了
    with amp_ctx():
        for b in dl:
            inputs = {
                "input_ids": b["input_ids"].to(device),
                "attention_mask": b["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            start_list.append(outputs.start_logits.detach().cpu())
            end_list.append(outputs.end_logits.detach().cpu())
            pbar.update(b["input_ids"].shape[0])
    pbar.close()

    start_logits = torch.cat(start_list, dim=0)  # [N_feat, L]
    end_logits = torch.cat(end_list, dim=0)      # [N_feat, L]

    # 建立 example_id -> feature indices 映射，以及 id -> context 快取
    ex2feat = defaultdict(list)
    id2ctx = {}
    for idx, (qid, _, ctx) in enumerate(records):
        id2ctx[qid] = ctx  # 後面以 SQuAD 為準，內容一致
    for i, qid in enumerate(feats["example_id"]):
        ex2feat[qid].append(i)

    # 後處理（逐題彙整所有切片，向量化暴力找最佳 span）
    def pick_best_span_for_example(qid: str) -> str:
        ctx = id2ctx[qid]
        best_text = ""
        best_score = float("-inf")

        for fi in ex2feat[qid]:
            offsets = feats["offset_mapping"][fi]
            s = start_logits[fi]
            e = end_logits[fi]

            # 有效 token（僅 context）
            valid_idx = [ti for ti, o in enumerate(offsets) if (o is not None and o[0] != o[1])]
            if not valid_idx:
                continue

            idx_v = torch.tensor(valid_idx, dtype=torch.long)
            s_v = s.index_select(0, idx_v)
            e_v = e.index_select(0, idx_v)

            # 向量化暴力：分數矩陣 M = s[:,None] + e[None,:]
            M = s_v[:, None] + e_v[None, :]

            # 遮罩：ei >= si 且 (ei - si + 1) <= max_answer_len
            si_pos = idx_v[:, None]
            ei_pos = idx_v[None, :]
            legal = (ei_pos >= si_pos) & ((ei_pos - si_pos + 1) <= max_answer_len)
            M = M.masked_fill(~legal, float("-inf"))

            # 取最大
            max_val, flat_idx = torch.max(M.view(-1), dim=0)
            if torch.isfinite(max_val):
                ns, ne = M.shape
                si_idx = int(flat_idx // ne)
                ei_idx = int(flat_idx % ne)
                si_orig = int(idx_v[si_idx].item())
                ei_orig = int(idx_v[ei_idx].item())
                stc, edc = offsets[si_orig][0], offsets[ei_orig][1]
                cand = ctx[stc:edc].strip()
                if cand and float(max_val) > best_score:
                    best_score = float(max_val)
                    best_text = cand

        return best_text

    rows: List[Tuple[str, str]] = []
    for qid, _, _ in tqdm(records, desc="Postprocess (examples)", unit="ex"):
        rows.append((qid, pick_best_span_for_example(qid)))

    # 輸出 CSV
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "answer"])
        for r in rows:
            w.writerow(r)

    print(f"[Infer] 共寫入 {len(rows)} 筆到 {out_csv_path}")
    return len(rows)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="一鍵轉SQuAD並批次推論")
    # I/O
    ap.add_argument("--test_json", default=r"C:/Users/USER/mnt/data/test.json")
    ap.add_argument("--context_json", default=r"C:/Users/USER/mnt/data/context.json")
    ap.add_argument("--mc_pred_json", default=r"C:/Users/USER/mnt/data/output/mc_predictions.json")
    ap.add_argument("--out_squad_json", default=r"C:/Users/USER/mnt/data/output/test_squad.json")
    ap.add_argument("--out_csv", default=r"C:/Users/USER/mnt/data/output/test_predictions.csv")

    # 模型與 tokenizer
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--tokenizer_name", default=None)

    # 推論超參數
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--doc_stride", type=int, default=128)
    ap.add_argument("--max_answer_len", type=int, default=64)

    # 批次與加速
    ap.add_argument("--batch", type=int, default=32, help="feature 批次大小（非題目數，指滑窗後的切片數）")
    ap.add_argument("--fp16", action="store_true", help="僅 GPU 有效，使用自動混合精度")

    # 離線模式
    ap.add_argument("--local_files_only", action="store_true", help="僅從本地載入模型/權重，不連網")
    ap.add_argument("--title", default="test")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) 轉 SQuAD
    build_squad_from_mc(
        test_json_path=args.test_json,
        context_json_path=args.context_json,
        mc_pred_json_path=args.mc_pred_json,
        out_squad_path=args.out_squad_json,
        title=args.title
    )

    # 2) 批次推論
    infer_qa_on_squad_batched(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name=args.tokenizer_name,
        test_squad_json_path=args.out_squad_json,
        out_csv_path=args.out_csv,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_answer_len=args.max_answer_len,
        batch=args.batch,
        fp16=args.fp16,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()

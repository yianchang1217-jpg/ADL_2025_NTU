# -*- coding: utf-8 -*-
# run_longformer_qa_allinone.py
import json, argparse, collections, os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from datasets import Dataset
#from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import AutoTokenizer, LongformerForQuestionAnswering, TrainingArguments, Trainer
# from tqdm.notebook import tqdm
# import transformers.trainer_utils as tu
# tu.tqdm = tqdm  # 強制 Trainer 使用 notebook-friendly 的進度條

# ========= 讀寫 =========
def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ========= 從 context.json 取段落內容 =========
def paragraphs_to_texts(par_ids, contexts):
    """
    支援三種 context.json 形態：
    1) list/array → 以整數索引
    2) dict（鍵是 int）
    3) dict（鍵是 str）
    """
    texts = []
    if isinstance(contexts, list):
        n = len(contexts)
        for pid in par_ids:
            idx = int(pid)
            if not (0 <= idx < n):
                raise KeyError(f"paragraph pid {pid} 超出 list 範圍 (0..{n-1})")
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
            raise KeyError(f"paragraph pid {pid} 找不到")
        return texts

    raise TypeError(f"context.json 型別不支援: {type(contexts)}")

def resolve_relevant_index(relevant, par_ids):
    if isinstance(relevant, int) and 0 <= relevant < len(par_ids):
        return relevant
    if relevant in par_ids:
        return par_ids.index(relevant)
    rel_str = str(relevant)
    par_ids_str = list(map(str, par_ids))
    if rel_str in par_ids_str:
        return par_ids_str.index(rel_str)
    raise ValueError(f"relevant={relevant} 無法在 paragraphs={par_ids} 中定位")

def build_big_context_and_offset(texts: List[str], rel_idx: int, sep: str) -> Tuple[str, int]:
    big = sep.join(texts)
    offset = sum(len(t) for t in texts[:rel_idx]) + rel_idx * len(sep)
    return big, offset

# ========= 將原始 split（train/valid）轉扁平 SQuAD =========
def convert_split_flat(
    split_data: List[Dict[str, Any]],
    contexts: Any,
    combine_mode: str = "big",
    sep: str = "\n\n"
) -> List[Dict[str, Any]]:
    """
    輸出每筆：id, question, context, answers:{text:[...], answer_start:[...]}
    combine_mode:
      - "big"      : 4 段用 sep 串成一段，並平移 start（等同你原本 build.py）
      - "relevant" : 只取正解段當 context，start 不平移
    """
    out_rows: List[Dict[str, Any]] = []
    for ex in split_data:
        qid = ex["id"]
        question = ex["question"]
        par_ids = ex["paragraphs"]
        relevant = ex["relevant"]
        ans = ex["answer"]  # {"text": str, "start": int}

        texts = paragraphs_to_texts(par_ids, contexts)
        rel_idx = resolve_relevant_index(relevant, par_ids)

        if combine_mode == "big":
            context, offset = build_big_context_and_offset(texts, rel_idx, sep)
            new_start = offset + int(ans["start"])
        else:  # "relevant"
            context = texts[rel_idx]
            new_start = int(ans["start"])

        ans_text = ans["text"]

        # 安全檢查
        if 0 <= new_start <= len(context):
            slice_txt = context[new_start:new_start + len(ans_text)]
            if slice_txt != ans_text:
                print(f"[WARN] id={qid}: 切片與答案不一致（old_start={ans['start']}, new_start={new_start}）")

        out_rows.append({
            "id": qid,
            "question": question,
            "context": context,
            "answers": {"text": [ans_text], "answer_start": [new_start]},
        })
    return out_rows

# ========= Longformer 前處理 =========
def prepare_features(examples, tokenizer, max_length=4096, doc_stride=256):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    # 建 example_id，並把非 context 的 offset 設成 None（只留 context）
    example_ids = []
    new_offsets = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx])
        seq_ids = tokenized.sequence_ids(i)
        temp = []
        for k, o in enumerate(offsets):
            temp.append(o if seq_ids[k] == 1 else None)
        new_offsets.append(temp)
    tokenized["example_id"] = example_ids
    tokenized["offset_mapping"] = new_offsets

    # 計算 start/end_positions（沒有答案時指到 CLS）
    start_positions, end_positions = [], []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
        sample_idx = sample_mapping[i]
        start_char = examples["answer_start"][sample_idx]
        end_char = start_char + len(examples["answer_text"][sample_idx])

        # 找 context 區段 token 邊界
        seq_ids = tokenized.sequence_ids(i)
        idx = 0
        while idx < len(seq_ids) and seq_ids[idx] != 1: idx += 1
        context_start = idx
        while idx < len(seq_ids) and seq_ids[idx] == 1: idx += 1
        context_end = idx - 1

        # 答案不在此切片 → 指到 CLS
        if not (offsets[context_start] and offsets[context_end] and
                offsets[context_start][0] <= start_char <= offsets[context_end][1] and
                offsets[context_start][0] <= end_char   <= offsets[context_end][1]):
            start_positions.append(cls_index); end_positions.append(cls_index)
            continue

        # 二分搜尋亦可；這裡線性掃
        s = context_start
        while s <= context_end and (offsets[s] is None or offsets[s][0] <= start_char): s += 1
        e = context_end
        while e >= context_start and (offsets[e] is None or offsets[e][1] >= end_char): e -= 1
        start_positions.append(s - 1); end_positions.append(e + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    # 手動提供 global_attention_mask：CLS + 問句 tokens 設為 1
    global_masks = []
    for i in range(len(tokenized["input_ids"])):
        seq_ids = tokenized.sequence_ids(i)
        m = [0] * len(tokenized["input_ids"][i])
        m[0] = 1
        for j, sid in enumerate(seq_ids):
            if sid == 0:  # question
                m[j] = 1
        global_masks.append(m)
    tokenized["global_attention_mask"] = global_masks
    return tokenized

# ========= 指標 =========
def compute_em_f1(preds: List[str], gts: List[str]):
    def norm(s): return s.strip()
    def f1(p, g):
        p, g = list(norm(p)), list(norm(g)); inter=0; used=[False]*len(g)
        for ch in p:
            for i, gh in enumerate(g):
                if not used[i] and ch==gh: used[i]=True; inter+=1; break
        prec = inter/len(p) if p else 0; rec = inter/len(g) if g else 0
        return 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    em = np.mean([float(norm(a)==norm(b)) for a,b in zip(preds,gts)])
    f1s = np.mean([f1(a,b) for a,b in zip(preds,gts)])
    return {"exact_match": float(em), "f1": float(f1s)}

# def postprocess_predictions(examples: Dataset, features: Dataset, raw_predictions, max_answer_len=64):
#     start_logits, end_logits = raw_predictions
#     # features → example_id 的反索引
#     f_per_ex = collections.defaultdict(list)
#     for i, ex_id in enumerate(features["example_id"]):
#         f_per_ex[ex_id].append(i)
#     # id → row index
#     id2idx = {id_: i for i, id_ in enumerate(examples["id"])}

#     final_texts = []
#     for ex_id in examples["id"]:
#         ctx = examples["context"][id2idx[ex_id]]
#         best_text = ""; best_score = -1e9
#         for fi in f_per_ex.get(ex_id, []):
#             offsets = features["offset_mapping"][fi]
#             s_log = start_logits[fi]; e_log = end_logits[fi]
#             for s in range(len(s_log)):
#                 if offsets[s] is None: continue
#                 e_max = min(len(e_log)-1, s+max_answer_len-1)
#                 for e in range(s, e_max+1):
#                     if offsets[e] is None: continue
#                     score = float(s_log[s]) + float(e_log[e])
#                     if score > best_score:
#                         start_char, end_char = offsets[s][0], offsets[e][1]
#                         best_text = ctx[start_char:end_char]
#                         best_score = score
#         final_texts.append(best_text.strip())
#     return final_texts
def postprocess_predictions(examples: Dataset, features: Dataset, raw_predictions, max_answer_len=64, topk=20):
    """
    以 Top-K start/end 縮小搜尋空間，顯著加速後處理。
    與原版相比：把 O(T * max_answer_len) 的雙層掃描，換成 O(topk^2) 的組合。
    """
    import numpy as np
    from tqdm import tqdm

    start_logits, end_logits = raw_predictions  # 兩個 shape: [num_features, seq_len]

    # features -> example_id 映射
    f_per_ex = collections.defaultdict(list)
    for i, ex_id in enumerate(features["example_id"]):
        f_per_ex[ex_id].append(i)

    # 方便查原始 context
    id2idx = {id_: i for i, id_ in enumerate(examples["id"])}

    final_texts = []
    for ex_id in tqdm(examples["id"], desc="Postprocess"):
        ctx = examples["context"][id2idx[ex_id]]
        best_text = ""
        best_score = -1e9

        for fi in f_per_ex.get(ex_id, []):
            offsets = features["offset_mapping"][fi]
            s_log = np.asarray(start_logits[fi])
            e_log = np.asarray(end_logits[fi])

            # 只保留屬於 context 的 token（offset != None）
            valid_idx = [i for i, off in enumerate(offsets) if off is not None]
            if not valid_idx:
                continue
            s_log_valid = s_log[valid_idx]
            e_log_valid = e_log[valid_idx]

            # 取 Top-K 的 index（在 valid 區間）
            k = min(topk, len(valid_idx))
            s_top_rel = s_log_valid.argsort()[-k:]  # 相對於 valid_idx 的位置
            e_top_rel = e_log_valid.argsort()[-k:]

            # 轉回絕對 token index
            s_top = [valid_idx[i] for i in s_top_rel]
            e_top = [valid_idx[i] for i in e_top_rel]

            # 嘗試配對
            for s in s_top:
                # 限制 e 在 [s, s+max_answer_len-1]
                e_max = min(len(e_log) - 1, s + max_answer_len - 1)
                for e in e_top:
                    if e < s or e > e_max:
                        continue
                    score = float(s_log[s]) + float(e_log[e])
                    if score > best_score:
                        start_char, end_char = offsets[s][0], offsets[e][1]
                        cand = ctx[start_char:end_char].strip()
                        # 避免取到空字串
                        if cand:
                            best_text = cand
                            best_score = score

        final_texts.append(best_text)

    return final_texts

# ========= 主流程 =========
def main():
    ap = argparse.ArgumentParser()
    # 1) 原始資料
    ap.add_argument("--base_dir", default=r"C:/Users/USER/mnt/data/bonus")
    ap.add_argument("--train_json", default="train.json")   # 或 train_pre5.json，請視你的檔名調整
    ap.add_argument("--valid_json", default="valid.json")
    ap.add_argument("--context_json", default="context.json")
    ap.add_argument("--combine_mode", choices=["big","relevant"], default="big")
    ap.add_argument("--sep", default="\n\n")
    ap.add_argument("--dump_intermediate", action="store_true",
                    help="如開啟，會把扁平資料也輸出成 *_squad_flat.json 方便檢查")

    # 2) 模型 / 訓練
    ap.add_argument("--model_name", default="ValkyriaLenneth/longformer_zh")
    ap.add_argument("--output_dir", default=r"C:/Users/USER/mnt/data/output/longformer_qa_3070")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--doc_stride", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    args = ap.parse_args()

    base = Path(args.base_dir)
    train_path = base / args.train_json
    valid_path = base / args.valid_json
    context_path = base / args.context_json
    os.makedirs(args.output_dir, exist_ok=True)

    # 讀原始 + 轉扁平
    contexts = load_json(context_path)
    train_raw = load_json(train_path)
    valid_raw = load_json(valid_path)

    train_flat = convert_split_flat(train_raw, contexts, args.combine_mode, args.sep)
    valid_flat = convert_split_flat(valid_raw, contexts, args.combine_mode, args.sep)

    if args.dump_intermediate:
        save_json({"data": train_flat}, base / "train_squad_flat.json")
        save_json({"data": valid_flat}, base / "valid_squad_flat.json")

    # 扁平 → HF Dataset
    def rows_to_ds(rows):
        return Dataset.from_dict({
            "id": [r["id"] for r in rows],
            "question": [r["question"] for r in rows],
            "context": [r["context"] for r in rows],
            "answer_text": [r["answers"]["text"][0] for r in rows],
            "answer_start": [int(r["answers"]["answer_start"][0]) for r in rows],
        })
    train_ds = rows_to_ds(train_flat)
    valid_ds = rows_to_ds(valid_flat)

    # Tokenize + 建 features（含 global_attention_mask）
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = LongformerForQuestionAnswering.from_pretrained(args.model_name)
    #model = transformers.LongformerModel.from_pretrained(args.model_name)
    print("torch.cuda.is_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
        print("bf16_supported =", torch.cuda.is_bf16_supported())

    train_feats = train_ds.map(
        lambda e: prepare_features(e, tokenizer, args.max_length, args.doc_stride),
        batched=True, remove_columns=train_ds.column_names
    )
    valid_feats = valid_ds.map(
        lambda e: prepare_features(e, tokenizer, args.max_length, args.doc_stride),
        batched=True, remove_columns=valid_ds.column_names
    )

    # 混合精度：優先 bf16（Ampere 起可用），否則 fp16；並關掉梯度裁剪避免與 GradScaler 互撞
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        max_grad_norm=0.0,         # 關閉梯度裁剪，避免 GradScaler unscale_ 相關錯誤
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        report_to=["none"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    def compute_metrics_fn(p):
        preds = postprocess_predictions(valid_ds, valid_feats, p.predictions, max_answer_len=64)
        gts = [t for t in valid_ds["answer_text"]]
        return compute_em_f1(preds, gts)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_feats,
        eval_dataset=valid_feats,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    print(">>> Predict on valid ...")
    # === 產生並列印驗證集預測 ===
    eval_preds = trainer.predict(valid_feats)  # 會得到 (start_logits, end_logits)

    print(">>> Postprocess predictions (這段是 CPU 密集、沒有 GPU 進度條)")
    pred_texts = postprocess_predictions(valid_ds, valid_feats, eval_preds.predictions, max_answer_len=64)

    # 印出前 N 筆（可自行調整 N）
    N = 5  # 或者改成 20
    for i, (ex_id, q, gt, pd) in enumerate(zip(
        valid_ds["id"], valid_ds["question"], valid_ds["answer_text"], pred_texts
    )):
        if i >= N: break
        print(f"[{i:04d}] id={ex_id}\nQ: {q}\nGT: {gt}\nPD: {pd}\n---")

    # # 存成 JSON 方便之後查看
    # pred_rows = []
    # for ex_id, q, ctx, gt, pd in zip(
    #     valid_ds["id"], valid_ds["question"], valid_ds["context"], valid_ds["answer_text"], pred_texts
    # ):
    #     pred_rows.append({"id": ex_id, "question": q, "context": ctx, "gold": gt, "pred": pd})
    # save_json(pred_rows, Path(args.output_dir) / "valid_predictions.json")

    # 原本就有的保存
    print(">>> Save model & tokenizer ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

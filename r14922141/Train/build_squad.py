# -*- coding: utf-8 -*-
"""
build_squad_flat_only.py

功能（只輸出 FLAT）：
- 一次處理 train 與 valid
- 不再用 Multiple-Choice 模型挑段，直接用每筆樣本的 `relevant` 欄位當作正解段落
- 只輸出「扁平化版本（每列一題 QA）」；不再輸出 SQuAD v1.1 結構
- 支援 CONTEXTS 為 list 或 dict（鍵可為字串或整數）
- 支援 relevant 為字串或整數
- 找不到答案時會跳過並統計
- 在每個 split 完成後，print 前 N 筆 QA 預覽

用法（Windows / Git Bash / Linux）：
python build_squad.py \
  --context_json "C:/Users/USER/mnt/data/context.json" \
  --train_in "C:/Users/USER/mnt/data/train.json" \
  --train_flat_out "C:/Users/USER/mnt/data/output/train_squad_flat.json" \
  --valid_in "C:/Users/USER/mnt/data/valid.json" \
  --valid_flat_out "C:/Users/USER/mnt/data/output/valid_squad_flat.json" \
  --preview_n 5
"""

import json
import uuid
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import argparse


# ============
# 小工具
# ============
def to_str(x: Any) -> str:
    """將值轉為字串；若為 list 取第一個；None 則回空字串。"""
    if x is None:
        return ""
    if isinstance(x, list):
        return str(x[0]) if x else ""
    return str(x)


def squeeze_ws(s: str) -> str:
    """壓縮連續空白為單一空白"""
    return re.sub(r"\s+", " ", s).strip()


def safe_find_answer(context: str, answer_text: str) -> int:
    """
    在 context 裡找 answer_text 的起點；容錯空白/大小寫，並嘗試將壓縮後的位置 map 回原文。
    回傳 -1 表示找不到。
    """
    # 直接找
    idx = context.find(answer_text)
    if idx != -1:
        return idx

    # 壓縮空白後找
    ctx_sq = squeeze_ws(context)
    ans_sq = squeeze_ws(answer_text)

    idx_sq = ctx_sq.find(ans_sq)
    if idx_sq == -1:
        # 不分大小寫
        idx_sq = ctx_sq.lower().find(ans_sq.lower())
        if idx_sq == -1:
            return -1

    # 把壓縮字串位置映回原 context
    def map_index(orig: str, sq: str, target_sq_idx: int) -> int:
        i = j = 0
        last_was_ws = False
        while i < len(orig) and j < len(sq):
            c = orig[i]
            if c.isspace():
                if not last_was_ws:
                    # sq 中以單一空白代表一串空白
                    if j == target_sq_idx:
                        return i
                    j += 1
                    last_was_ws = True
            else:
                if j == target_sq_idx:
                    return i
                j += 1
                last_was_ws = False
            i += 1
        return i if j == target_sq_idx else -1

    return map_index(context, ctx_sq, idx_sq)


def get_context_by_rel(contexts: Union[List[str], Dict[Any, str]], rel_idx: Any) -> str:
    """
    依 rel_idx 取出 context 文字。contexts 可為 list 或 dict（鍵可為 str/int）。
    rel_idx 可為 str/int；若為 str 會先嘗試轉 int。
    """
    # 嘗試整數索引
    try:
        idx_int = int(rel_idx)
    except Exception:
        idx_int = None

    if isinstance(contexts, list):
        if idx_int is None or idx_int < 0 or idx_int >= len(contexts):
            raise KeyError(f"relevant={rel_idx} 超出 contexts(list) 範圍 0..{len(contexts)-1}")
        return to_str(contexts[idx_int])

    if isinstance(contexts, dict):
        # 先試原值
        if rel_idx in contexts:
            return to_str(contexts[rel_idx])
        # 再試字串鍵
        k_str = str(rel_idx)
        if k_str in contexts:
            return to_str(contexts[k_str])
        # 再試整數鍵
        if idx_int is not None and idx_int in contexts:
            return to_str(contexts[idx_int])

    raise KeyError(f"relevant={rel_idx} 在 contexts 中找不到對應項目。")


def flatten_file_from_data(src: Dict[str, Any], path_out: str) -> None:
    """將「SQuAD 結構的中介物件」轉為扁平陣列格式（每列一個 QA），並直接寫出檔案。"""
    rows: List[Dict[str, Any]] = []
    for art in src.get("data", []):
        title = art.get("title", "")
        for para in art.get("paragraphs", []):
            ctx = para.get("context", "")
            for qa in para.get("qas", []):
                qa_id = qa.get("id") or str(uuid.uuid4())
                ans = qa.get("answers", [])
                if isinstance(ans, dict):
                    ans = [ans]
                answers = {
                    "text": [a.get("text", "") for a in ans],
                    "answer_start": [a.get("answer_start", -1) for a in ans],
                }
                rows.append({
                    "id": qa_id,
                    "title": title,
                    "context": ctx,
                    "question": qa.get("question", ""),
                    "answers": answers
                })

    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump({"data": rows}, f, ensure_ascii=False, indent=2)


def process_split(
    split_name: str,
    in_path: str,
    out_flat: str,
    title: str,
    contexts: Union[List[str], Dict[Any, str]],
    preview_n: int = 5
) -> Tuple[int, int]:
    """
    讀入 split 原始 JSON（需包含 question/relevant/answer 或 answer_text），
    建立中介 SQuAD 結構於記憶體，最後僅輸出扁平版本；回傳 (kept, skipped)。
    """
    print(f"\n=== Processing split: {split_name} ===")
    print(f"Input: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 先累積為 SQuAD 結構的中介物件（只存在記憶體，不落地）
    data_obj: Dict[str, Any] = {
        "version": "v1.1",
        "data": [{"title": title, "paragraphs": []}]
    }
    paragraphs = data_obj["data"][0]["paragraphs"]

    cnt_ok = 0
    cnt_skip = 0

    for ex in data:
        question = to_str(ex.get("question", ""))
        rel = ex.get("relevant", None)
        try:
            rel_idx = int(rel)  # 優先用 int
        except Exception:
            rel_idx = rel

        # 取 context
        try:
            context = get_context_by_rel(contexts, rel_idx)
        except Exception:
            cnt_skip += 1
            continue

        # 取答案文字：支援 {"answer":{"text":..}} 或 "answer_text"
        raw_ans = ex.get("answer", None)
        if isinstance(raw_ans, dict) and "text" in raw_ans:
            answer_text = to_str(raw_ans.get("text", ""))
        else:
            answer_text = to_str(ex.get("answer_text", ""))

        if not answer_text:
            cnt_skip += 1
            continue

        start = safe_find_answer(context, answer_text)
        if start == -1:
            cnt_skip += 1
            continue

        qid = ex.get("id") or str(uuid.uuid4())
        qa_item = {
            "id": qid,
            "question": question,
            "answers": [{"text": answer_text, "answer_start": int(start)}]
        }
        paragraph_item = {"context": context, "qas": [qa_item]}
        paragraphs.append(paragraph_item)
        cnt_ok += 1

    # 預覽前 N 筆
    print("\nPreview first {} QAs:".format(preview_n))
    for para in paragraphs[:preview_n]:
        ctx_preview = para["context"][:80].replace("\n", " ")
        print("Context:", ctx_preview, "...")
        for qa in para["qas"]:
            print("  Q:", qa["question"])
            print("  A:", qa["answers"])

    # 只輸出 FLAT
    flatten_file_from_data(data_obj, out_flat)
    print(f"[FLAT] {split_name}: {out_flat} | kept={cnt_ok}, skipped={cnt_skip}")

    return cnt_ok, cnt_skip


# ============
# CLI
# ============
def parse_args():
    ap = argparse.ArgumentParser(description="用 relevant 欄位建立 train/valid 的扁平 FLAT 檔（不輸出 SQuAD 檔）")
    # 預設路徑（可依需求覆蓋）
    ap.add_argument("--context_json", default=r"C:/Users/USER/mnt/data/context.json")

    ap.add_argument("--train_in", default=r"C:/Users/USER/mnt/data/train.json")
    ap.add_argument("--train_flat_out", default=r"C:/Users/USER/mnt/data/output/train_squad_flat.json")
    ap.add_argument("--train_title", default="from_relevant_train")

    ap.add_argument("--valid_in", default=r"C:/Users/USER/mnt/data/valid.json")
    ap.add_argument("--valid_flat_out", default=r"C:/Users/USER/mnt/data/output/valid_squad_flat.json")
    ap.add_argument("--valid_title", default="from_relevant_valid")

    ap.add_argument("--preview_n", type=int, default=5, help="每個 split 預覽筆數")
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Loading contexts: {args.context_json}")
    with open(args.context_json, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    # train -> 只輸出 flat
    process_split(
        split_name="train",
        in_path=args.train_in,
        out_flat=args.train_flat_out,
        title=args.train_title,
        contexts=contexts,
        preview_n=args.preview_n
    )

    # valid -> 只輸出 flat
    process_split(
        split_name="valid",
        in_path=args.valid_in,
        out_flat=args.valid_flat_out,
        title=args.valid_title,
        contexts=contexts,
        preview_n=args.preview_n
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()

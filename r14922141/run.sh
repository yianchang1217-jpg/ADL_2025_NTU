#!/bin/bash

CONTEXT_JSON=$1
TEST_JSON=$2
PRED_CSV=$3

# Step 1: Multiple-choice inference
python infer_mc.py \
  --model_dir lert_mc \
  --test_json "$TEST_JSON" \
  --context_json "$CONTEXT_JSON" \
  --out_json mc_lert_pred.json \
  --batch_size 32 \
  --max_length 512 \
  --fp16

# Step 2: QA inference
python infer_qa.py \
  --model_name_or_path pert_qa_large_10epo \
  --test_json "$TEST_JSON" \
  --context_json "$CONTEXT_JSON" \
  --mc_pred_json mc_lert_pred.json \
  --out_squad_json test_squad.json \
  --out_csv "$PRED_CSV" \
  --max_seq_length 512 \
  --doc_stride 128 \
  --batch 32 \
  --fp16 \
  --max_answer_len 64

#bash ./run.sh ./data/context.json ./data/test.json ./prediction.csv
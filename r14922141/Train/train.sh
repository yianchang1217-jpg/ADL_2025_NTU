python ./run_swag_no_trainer.py \
  --train_file "./data/train.json" \
  --validation_file "./data/valid.json" \
  --model_name_or_path "hfl/chinese-lert-base" \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --output_dir "./output/mc_output_lert"

python ./build_squad.py \
  --context_json "./data/context.json" \
  --train_in "./data/train.json" \
  --train_flat_out "./output/train_squad.json" \
  --valid_in "./data/valid.json" \
  --valid_flat_out "./output/valid_squad.json" 

python ./run_qa_no_trainer.py \
  --train_file "./output/train_squad.json" \
  --validation_file "./output/valid_squad.json" \
  --model_name_or_path "hfl/chinese-pert-large" \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 1 \
  --num_warmup_steps 1500 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --output_dir "./output/pert_qa_large_10epo" \
  --with_tracking

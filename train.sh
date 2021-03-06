#!/usr/bin/env bash

BERT_DIR=bleurt/zh_checkpoint/chinese_L-12_H-768_A-12
BERT_CKPT=bleurt/zh_checkpoint/chinese_L-12_H-768_A-12/bert_model.ckpt
python3 -m bleurt.finetune \
  -init_checkpoint=${BERT_CKPT} \
  -bert_config_file=${BERT_DIR}/bert_config.json \
  -vocab_file=${BERT_DIR}/vocab.txt \
  -model_dir=bleurt_checkpoint \
  -train_set=pretrain/data_generated/rating_train.jsonl \
  -dev_set=pretrain/data_generated/rating_dev.jsonl \
  -num_train_steps=5000000

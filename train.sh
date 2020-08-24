#!/usr/bin/env bash

BERT_DIR=bleurt/zh_checkpoint
BERT_CKPT=multi_cased_L-12_H-768_A-12/bert_model.ckpt
python3 -m bleurt.finetune \
  -init_checkpoint=${BERT_DIR}/${BERT_CKPT} \
  -bert_config_file=${BERT_DIR}/bert_config.json \
  -vocab_file=${BERT_DIR}/vocab.txt \
  -model_dir=my_new_bleurt_checkpoint \
  -train_set=pretrain/data_generationed/ratings_train.jsonl \
  -dev_set=pretrain/data_generationed/ratings_dev.jsonl \
  -num_train_steps=500000

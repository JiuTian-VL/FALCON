#!/bin/bash

# Dataset choices
# [DocVQA, InfographicsVQA, WikiTableQuestions, DeepForm,
# KleisterCharity, TabFact, ChartQA, TextVQA, TextCaps, VisualMRC]

CKPT_NAME='jiutian'
MODEL_PATH='/path/to/jiutian-falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_2'

DATASET='InfographicsVQA'
DATA_DIR='/data2/Datasets/DocDownstream-1.0'
OUTPUT_DIR="./outputs/eval/$DATASET"

python -m jiutian.eval.model_eval_doc \
  --model-path $MODEL_PATH \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --ckpt_name $CKPT_NAME \
  --conv-mode $CONV_MODE
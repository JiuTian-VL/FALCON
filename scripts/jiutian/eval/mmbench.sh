#!/bin/bash

SPLIT="mmbench_test_en_20231003"

CKPT_NAME='jiutian'
MODEL_PATH='/path/to/jiutian-falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

ANNOTATION_FILE="/data/eval/mmbench/$SPLIT.tsv"
OUTPUT_DIR='./outputs/eval/mmbench'

python -m jiutian.eval.model_eval_mmbench \
    --model-path $MODEL_PATH \
    --question-file $ANNOTATION_FILE \
    --answers-file $OUTPUT_DIR/answers/$SPLIT/$CKPT_NAME.jsonl \
    --single-pred-prompt \
    --conv-mode $CONV_MODE

mkdir -p $OUTPUT_DIR/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $ANNOTATION_FILE \
    --result-dir $OUTPUT_DIR/answers/$SPLIT \
    --upload-dir $OUTPUT_DIR/answers_upload/$SPLIT \
    --experiment $CKPT_NAME

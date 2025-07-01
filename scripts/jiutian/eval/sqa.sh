#!/bin/bash

CKPT_NAME='jiutian'
MODEL_PATH='/path/to/jiutian-falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

QUESTION_FILE='/data/eval/scienceqa/llava_test_CQM-A.json'
IMAGE_FOLDER='/data2/Datasets/scienceqa/images/test'

OUTPUT_DIR='./outputs/eval/scienceqa'
RESULT_FILE="./outputs/eval/scienceqa/$CKPT_NAME.jsonl"


python -m jiutian.eval.model_eval_science \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $OUTPUT_DIR/$CKPT_NAME.jsonl \
    --single-pred-prompt \
    --conv-mode $CONV_MODE

python evaluation/eval_science_qa.py \
    --base-dir /data2/Datasets/scienceqa \
    --result-file $RESULT_FILE \
    --output-file $OUTPUT_DIR/$CKPT_NAME_output.jsonl \
    --output-result $OUTPUT_DIR/$CKPT_NAME_result.jsonl

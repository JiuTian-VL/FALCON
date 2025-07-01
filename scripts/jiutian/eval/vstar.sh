#!/bin/bash

CKPT_NAME='jiutian'
MODEL_PATH='/path/to/jiutian-falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

BENCHMARK_FOLDER='/data2/Datasets/vstar_bench'

OUTPUT_DIR='./outputs/eval/vstar'
RESULT_FILE="./outputs/eval/vstar/$CKPT_NAME.jsonl"


python -m jiutian.eval.model_eval_vstar \
    --model-path $MODEL_PATH \
    --benchmark-folder $BENCHMARK_FOLDER \
    --answers-file $OUTPUT_DIR/$CKPT_NAME.jsonl \
    --conv-mode $CONV_MODE

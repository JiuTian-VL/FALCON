#!/bin/bash

MODEL_PATH='/path/to/jiutian-falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

QUESTION_FILE='/data/eval/pope/llava_pope_test.jsonl'
IMAGE_FOLDER='/data2/Datasets/coco/images/val2014'
POPE_DIR='/data2/Datasets/pope'

RESULT_FILE='./outputs/eval/pope/jiutian.jsonl'

python -m jiutian.eval.model_eval \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $RESULT_FILE \
    --conv-mode $CONV_MODE

python evaluation/eval_pope.py \
    --annotation-dir $POPE_DIR \
    --question-file $QUESTION_FILE \
    --result-file $RESULT_FILE

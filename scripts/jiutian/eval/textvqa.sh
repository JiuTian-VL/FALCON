#!/bin/bash

MODEL_PATH='/path/to/falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

QUESTION_FILE='../../llava_textvqa_val_v051.jsonl'   # Note that we remove the ocr input
ANNOTATION_FILE='/data2/Datasets/TextVQA/TextVQA_0.5.1_val.json'
IMAGE_FOLDER='/data2/Datasets/TextVQA/train_images'

RESULT_FILE='./outputs/eval/textvqa/jiutian.jsonl'

python -m jiutian.eval.model_eval \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file $RESULT_FILE \
    --conv-mode $CONV_MODE

python -m evaluation.eval_textvqa \
    --annotation-file $ANNOTATION_FILE \
    --result-file $RESULT_FILE

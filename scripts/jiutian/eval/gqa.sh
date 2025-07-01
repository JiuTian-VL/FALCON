#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="jiutian"
SPLIT="llava_gqa_testdev_balanced"

MODEL_PATH='/path/to/falcon8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

QUESTION_FILE="/data/eval/gqa/$SPLIT.jsonl"
GQA_DIR="/data2/Datasets/gqa"

OUTPUT_DIR='/data7/Users/zrs/codes/FALCON/outputs/eval/gqa'  # IMPORTANT: should be absolute path here

# enable ctrl c
trap 'echo "Terminating all processes..."; kill 0' SIGINT

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m jiutian.eval.model_eval \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $GQA_DIR/images \
        --answers-file $OUTPUT_DIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=$OUTPUT_DIR/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $OUTPUT_DIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $OUTPUT_DIR/answers/$SPLIT/$CKPT/testdev_balanced_predictions.json

cd $GQA_DIR
python eval/eval.py --tier testdev_balanced --predictions $OUTPUT_DIR/answers/$SPLIT/$CKPT/testdev_balanced_predictions.json

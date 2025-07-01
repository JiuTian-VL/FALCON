#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_NAME='jiutian'
MODEL_PATH='/path/to/jiutian-falcon-8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

QUESTION_FILE='/data2/Datasets/MME-RealWorld/MME_RealWorld.json'
IMAGE_FOLDER='/data2/Datasets/MME-RealWorld'

OUTPUT_DIR='./outputs/eval/mme_realworld'

# enable ctrl c
trap 'echo "Terminating all processes..."; kill 0' SIGINT

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m jiutian.eval.model_eval_mme_realworld \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $OUTPUT_DIR/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl \
        --conv-mode $CONV_MODE \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=$OUTPUT_DIR/$CKPT_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $OUTPUT_DIR/$CKPT_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python evaluation/eval_mme_realworld.py \
    --results_file $output_file

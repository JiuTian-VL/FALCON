#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH='/path/to/falcon8b'
#MODEL_BASE='/path/to/base'
CONV_MODE='llama_3_1'

QUESTION_FILE='/data/eval/seed_bench/llava-seed-bench-img.jsonl'
ANNOTATION_FILE='/data2/Datasets/seed/SEED-Bench.json'
IMAGE_FOLDER='/data2/Datasets/seed'

OUTPUT_DIR='./outputs/eval/seed_bench'

# enable ctrl c
trap 'echo "Terminating all processes..."; kill 0' SIGINT

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m jiutian.eval.model_eval \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $OUTPUT_DIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=$OUTPUT_DIR/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $OUTPUT_DIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file $ANNOTATION_FILE \
    --result-file $output_file \
    --result-upload-file $OUTPUT_DIR/answers_upload/$CKPT.jsonl


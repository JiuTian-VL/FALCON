import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
import random

from jiutian.constants import DEFAULT_IMAGE_TOKEN
from jiutian.utils import disable_torch_init
from jiutian.mm_utils import tokenizer_image_token, get_model_name_from_path
from jiutian.eval.model_infer import JiutianHDInfer
from .utils import extract_pred_option_regex


random.seed(0)
def shuffle_options(options):
    indices = list(range(len(options)))
    random.shuffle(indices)
    shuffled = [options[i] for i in indices]
    gt_idx = indices.index(0)
    return shuffled, gt_idx


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    model_infer = JiutianHDInfer(
        model_path=args.model_path,
        model_base=args.model_base,
        anchors='grid_9',
        add_global_img=True,
        add_textual_crop_indicator=True,
        conv_mode=args.conv_mode,
    )

    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []

    for test_type in ['direct_attributes', 'relative_position']:
        results[test_type] = []
        folder = os.path.join(args.benchmark_folder, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))

        for image_file in tqdm(image_files):
            result_single_sample = {}

            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'

            annotation = json.load(open(annotation_path))
            question = annotation['question']
            question = DEFAULT_IMAGE_TOKEN + '\n' + question

            # predict the multiple-choice option
            options = annotation['options']
            options, gt_idx = shuffle_options(options)
            gt = chr(gt_idx + 65)

            for idx, opt in enumerate(options):
                question += f'\n{chr(idx + 65)}. {opt}'
            question += "\nAnswer with the option’s letter from the given choices directly."

            image = Image.open(image_path).convert('RGB')
            option_chosen = model_infer.inference(image, question)
            option_chosen = extract_pred_option_regex(option_chosen)
            correct = 1 if option_chosen == gt else 0

            per_type_acc[test_type].append(correct)
            all_acc.append(correct)

            result_single_sample['question'] = question
            result_single_sample['options'] = options
            result_single_sample['image'] = image_file
            result_single_sample['option_chosen'] = option_chosen
            result_single_sample['correct'] = correct
            results[test_type].append(result_single_sample)

        print(test_type, np.mean(per_type_acc[test_type]))

    print(np.mean(all_acc))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(args.answers_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--benchmark-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")

    args = parser.parse_args()
    eval_model(args)
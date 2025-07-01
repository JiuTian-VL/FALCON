import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from icecream import ic

from jiutian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from jiutian.conversation import conv_templates
from jiutian.model.builder import load_pretrained_model
from jiutian.utils import disable_torch_init
from jiutian.mm_utils import tokenizer_image_token, get_model_name_from_path
from jiutian.eval.model_infer import JiutianHDInfer
from jiutian.eval.utils import get_chunk, read_jsonl, save_jsonl

from evaluation.eval_doc_benchmarks import (llm_text_localization_eval, llm_textcaps_textvqa_eval,llm_benchmark_eval)
from evaluation.eval_due_benchmarks import llm_duebenchmark_eval


def eval_model(args):
    # Model
    disable_torch_init()

    data_dir = args.data_dir
    dataset = args.dataset

    test_path = os.path.join(data_dir, 'test', dataset + '_test.jsonl')
    output_path = os.path.join(args.output_dir,  args.ckpt_name + '_test_pred.jsonl')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(output_path + ' exists, skip inference. ')
    else:
        model_name = get_model_name_from_path(args.model_path)
        model_infer = JiutianHDInfer(
            model_path=args.model_path,
            model_base=args.model_base,
            anchors='grid_9',
            add_global_img=True,
            add_textual_crop_indicator=True,
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            top_p=args.top_p,
            conv_mode=args.conv_mode,
        )

        test_samples = read_jsonl(test_path)
        infer_results = []
        for sample in tqdm(test_samples):
            image = os.path.join(data_dir, sample['image'][0])
            assert os.path.exists(image)

            question = sample['messages'][0]
            answer = sample['messages'][1]
            assert question['role'] == 'user'
            assert answer['role'] == 'assistant'

            query = DEFAULT_IMAGE_TOKEN + '\n' + question['content'].replace('<|image|>', '')
            query = query + '\nAnswer the question with a single word or phrase.'
            gt_answer = answer['content']

            model_answer = model_infer.inference(image, query)

            sample['model_answer'] = model_answer
            sample['gt_answer'] = gt_answer

            # ic(model_answer, gt_answer)
            infer_results.append(sample)

        save_jsonl(infer_results, output_path)

    if not os.path.exists(output_path):
        print('not exists:', output_path)
        exit(0)

    meta_dir = os.path.join(data_dir, 'meta')

    if dataset in ['DeepForm', 'DocVQA', 'InfographicsVQA', 'KleisterCharity', 'WikiTableQuestions']:
        llm_duebenchmark_eval(dataset_name=dataset, split='test', llm_pred_path=output_path, meta_dir=meta_dir)
    elif dataset in ['TabFact']:
        llm_benchmark_eval(metric_names=['ExactAccuracy'], result_path=output_path, save_each_eval=True)
    elif dataset in ['ChartQA']:
        llm_benchmark_eval(metric_names=['RelaxedAccuracy'], result_path=output_path, save_each_eval=True)
    elif dataset in ['TextCaps', 'TextVQA']:
        llm_textcaps_textvqa_eval(result_path=output_path, dataset=dataset, split='test', meta_dir=meta_dir)
    elif dataset in ['VisualMRC']:
        llm_benchmark_eval(
            metric_names=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'Meteor', 'RougeL', 'CIDEr'],
            result_path=output_path,
            save_each_eval=True
        )

    print('==============================================')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--ckpt_name", type=str, default="jiutian")
    parser.add_argument('--dataset', type=str,
                        choices=['DocVQA', 'InfographicsVQA', 'WikiTableQuestions',
                                 'DeepForm', 'KleisterCharity', 'TabFact', 'ChartQA',
                                 'TextVQA', 'TextCaps', 'VisualMRC'])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

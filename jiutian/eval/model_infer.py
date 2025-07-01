import argparse
import torch
from PIL import Image
from transformers import TextStreamer
import os
import copy
from icecream import ic
import time

from jiutian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from jiutian.conversation import conv_templates, SeparatorStyle
from jiutian.model.builder import load_pretrained_model
from jiutian.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_video_with_decord
from jiutian.processor import AdaptiveCropProcessor


class JiutianHDInfer():
    def __init__(
        self,
        model_path,
        model_base=None,
        anchors='grid_9',
        add_global_img=True,
        add_textual_crop_indicator=True,
        load_8bit=False,
        load_4bit=False,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=512,
        num_beams=1,
        top_p=1.0,
        conv_mode='v1',
    ):
        model_name = get_model_name_from_path(model_path)

        ic(model_name)
        ic(model_path)
        ic(model_base)
        ic(conv_mode)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name, load_8bit=load_8bit, load_4bit=load_4bit, device="cuda"
        )
        # self.image_processor = AdaptiveCropProcessor(
        #     image_size=336,
        #     anchors=anchors,
        #     add_global_img=add_global_img,
        #     add_textual_crop_indicator=add_textual_crop_indicator,
        #     enable_low_res=True
        # )
        ic(self.image_processor.anchors)

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.top_p = top_p
        self.conv_mode = conv_mode

        self.video_fps = 12
        self.frames_upbound = 16

    def inference(self, image, query):
        if image is not None:
            assert '<image>' in query
            processed_data = self.image_processor(images=image, query=query)

            image_tensor = processed_data['cropped_images']
            patch_positions = processed_data['patch_positions']
            text = processed_data['text']

            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            patch_positions = patch_positions.to(self.model.device, dtype=torch.long)
        else:
            assert '<image>' not in query
            text = query
            image_tensor = None
            patch_positions = None

        # ic(image_tensor.shape, patch_positions.shape, text)

        # conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv = conv_templates[self.conv_mode].copy()
        conv.tokenizer = self.tokenizer
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ic(prompt)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)

        # ic(input_ids)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        num_images = None if patch_positions is None else [patch_positions.shape[0]]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                num_images=num_images,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                # streamer=self.streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,  # llama3 series
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return outputs.replace('</s>', '')

    @torch.inference_mode()
    def likelihood_multi_choices_infer(self, image, query, options):
        if image is not None:
            assert '<image>' in query
            processed_data = self.image_processor(images=image, query=query)

            image_tensor = processed_data['cropped_images']
            patch_positions = processed_data['patch_positions']
            text = processed_data['text']

            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            patch_positions = patch_positions.to(self.model.device, dtype=torch.long)
        else:
            assert '<image>' not in query
            text = query
            image_tensor = None
            patch_positions = None

        # ic(image_tensor.shape, patch_positions.shape, text)

        # conv = copy.deepcopy(conv_templates[self.conv_mode])
        conv = conv_templates[self.conv_mode].copy()
        conv.tokenizer = self.tokenizer
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ic(prompt)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.model.device)

        # ic(input_ids)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_question = self.model(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                num_images=[patch_positions.shape[0]],
                use_cache=True,
            )

        question_logits = output_question.logits
        question_past_key_values = output_question.past_key_values

        loss_list = []

        for option in options:
            # conv = copy.deepcopy(conv_templates[self.conv_mode])
            conv = conv_templates[self.conv_mode].copy()
            conv.tokenizer = self.tokenizer
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], option)
            full_prompt = conv.get_prompt()

            full_input_ids = tokenizer_image_token(
                full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()
            option_answer_input_ids = full_input_ids[:, input_ids.shape[1]:]

            output_option = self.model(
                input_ids=option_answer_input_ids,
                attention_mask=torch.ones(
                    1, question_logits.shape[1] + option_answer_input_ids.shape[1], device=full_input_ids.device
                ),
                use_cache=True,
                past_key_values=question_past_key_values
            )

            logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

            loss_fct = torch.nn.CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = option_answer_input_ids.view(-1)
            loss = loss_fct(logits, labels)

            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model_infer = JiutianHDInfer(
        model_path=args.model_path,
        model_base=args.model_base,
        do_sample=False,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        top_p=args.top_p,
        conv_mode=args.conv_mode,
    )

    while(True):
        try:
            print('====================================')
            image_file = input("Image Path: ")
            question = input("Question: ")

            image_file = image_file.strip()
            question = question.strip()

            if '<image>' not in question:
                question = '<image>\n' + question

            st = time.time()
            answer = model_infer.inference(image_file, question)
            ed = time.time()

            cost_seconds = ed - st
            ic(cost_seconds)
            ic(answer)
        except Exception as e:
            print(e)
            continue



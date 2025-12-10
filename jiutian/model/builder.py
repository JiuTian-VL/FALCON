#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import CLIPImageProcessor

import torch
from jiutian.model import *
from jiutian.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from jiutian.processor import AdaptiveCropProcessor


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'jiutian' in model_name.lower() or 'falcon' in model_name.lower():
        # Load Jiutian model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        if 'lora' in model_name.lower() and model_base is not None:
            from jiutian.model.modeling_jiutian import JiutianConfig
            lora_cfg_pretrained = JiutianConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading Jiutian from base model...')
            model = JiutianLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional Jiutian weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            msg = model.load_state_dict(non_lora_trainables, strict=False)
            print("non_lora_trainables.bin Unexpected_keys:", msg.unexpected_keys)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading Jiutian from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = JiutianLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            weights = torch.load(os.path.join(model_path, 'model_trainables.bin'), map_location='cpu')
            weights = {k: v.to(torch.float16) for k, v in weights.items()}
            msg = model.load_state_dict(weights, strict=False)
            print('unexpected_keys:', msg.unexpected_keys)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = JiutianLlamaForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                **kwargs
            )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # image_processor = CLIPImageProcessor.from_pretrained(model.config.visual_config['model_name_or_path'])

    processor_config = model.config.processor_config
    image_processor = AdaptiveCropProcessor(
        image_size=processor_config["image_size"],
        anchors=processor_config["anchors"],
        add_global_img=processor_config["add_global_img"],
        add_textual_crop_indicator=processor_config["add_textual_crop_indicator"],
        enable_low_res=processor_config["enable_low_res"],
    )

    if 'jiutian' in model_name.lower():
        vision_model = model.get_vision_model()
        if not vision_model.is_loaded:
            vision_model.load_model()
        vision_model.to(device=device, dtype=torch.float16)

        # ========================================
        # TODO: hack -- load pretrained weights of ViT,
        #  as a workaround, the `model_trainables.bin` is manually copied
        def get_w(weights, keyword):
            return {k.split(keyword + '.', 1)[1]: v for k, v in weights.items() if keyword in k}

        if os.path.exists(os.path.join(model_path, 'model_trainables.bin')):
            weights = torch.load(os.path.join(model_path, 'model_trainables.bin'), map_location='cpu')
            weights = {k: v.to(torch.float16) for k, v in weights.items()}
            msg = model.get_model().get_vision_model().load_state_dict(get_w(weights, "vision_model"), strict=False)
            print('unexpected_keys:', msg.unexpected_keys)
        # ========================================

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

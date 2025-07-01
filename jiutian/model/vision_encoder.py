import os
from typing import Optional, Union, Tuple
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open
from transformers import CLIPVisionModel, CLIPVisionConfig, AutoModel
from transformers import SiglipVisionModel, SiglipVisionConfig


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""
#
#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         print(orig_type)
#         print(x.type(torch.float32).dtype)
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

class JiutianVisionModel(nn.Module):
    def __init__(self, config, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.config = config

        if 'clip' in self.config.model_name_or_path:
            self.vision_backbone = 'clip'
        else:
            self.vision_backbone = 'siglip'

        if self.vision_backbone == 'clip':
            print("Using 'CLIP' backbone ...")
            self.vision_tower_config = CLIPVisionConfig.from_pretrained(
                self.config.model_name_or_path,
            )
        elif self.vision_backbone == 'siglip':
            print("Using 'SIGLIP' backbone ...")
            self.vision_tower_config = SiglipVisionConfig.from_pretrained(
                self.config.model_name_or_path,
            )
        else:
            raise NotImplementedError

        self.vision_tower_config.num_vision_queries = config.num_vision_queries
        self.vision_tower_config.anchor_max = config.anchor_max
        self.vision_tower_config.enable_interactive_attn = config.enable_interactive_attn

        # Pretrained models that are not initialized
        # by `JiutianForCasualLM.from_pretrained()` should be loaded with a delay
        if not delay_load:
            self.load_model(load_hf_pretrained=False)

        hidden_size = self.vision_tower_config.hidden_size
        self.vision_queries = nn.Parameter(
            torch.randn(1, config.num_vision_queries, hidden_size)
        )

        self.crop_embedding_h = torch.nn.Embedding(config.anchor_max + 1, hidden_size)
        self.crop_embedding_w = torch.nn.Embedding(config.anchor_max + 1, hidden_size)

    def load_model(self, load_hf_pretrained=True, attn_implementation='eager', device_map=None):
        if self.is_loaded:
            print(f'`JiutianVisionModel.load_model` called again, skipping.')
            return

        print(f'======== vision load from HF: {load_hf_pretrained} ========')
        if load_hf_pretrained:
            # copied_weights = {}
            # state_dict = torch.load(os.path.join(self.config.model_name_or_path, 'pytorch_model.bin'))
            #
            # if self.vision_tower_config.enable_interactive_attn:
            #     for name, param in state_dict.items():
            #         if 'self_attn' in name:
            #             copied_weights[name.replace('self_attn', 'interact_attn')] = param.clone()
            #     state_dict.update(copied_weights)
            #
            # self.vision_tower = CLIPVisionModel.from_pretrained(
            #     self.config.model_name_or_path,
            #     config=self.vision_tower_config,
            #     state_dict=state_dict,
            #     attn_implementation=attn_implementation
            # )

            if self.vision_backbone == 'clip':
                self.vision_tower, msg = CLIPVisionModel.from_pretrained(
                    self.config.model_name_or_path,
                    config=self.vision_tower_config,
                    attn_implementation=attn_implementation,
                    output_loading_info=True,
                )
            elif self.vision_backbone == 'siglip':
                self.vision_tower, msg = SiglipVisionModel.from_pretrained(
                    self.config.model_name_or_path,
                    config=self.vision_tower_config,
                    attn_implementation=attn_implementation,
                    output_loading_info=True,
                )
            else:
                raise NotImplementedError

            if self.config.enable_interactive_attn and any('interact_attn' in k for k in msg['missing_keys']):
                print("========= Copying parameters from self-attn to interct-attn =========")
                for layer in self.vision_tower.vision_model.encoder.layers:
                    layer.interact_attn.load_state_dict(layer.self_attn.state_dict())

        else:
            # Note: transformers < 4.36 does not support CLIPVisionModel for AutoModel
            self.vision_tower_config._attn_implementation = attn_implementation
            self.vision_tower = AutoModel.from_config(self.vision_tower_config)
        self.is_loaded = True

    def feature_select_wo_cls(self, image_features, feature_type='patch'):
        if feature_type == 'query':
            selected_features = image_features[:, -self.config.num_vision_queries:, :]
        elif feature_type == 'patch':
            selected_features = image_features[:, :-self.config.num_vision_queries, :]
        else:
            selected_features = image_features

        return selected_features

    def feature_select(self, image_features, feature_type='patch'):
        # ========= traditional method =========
        # for traditional method
        if image_features.shape[1] == 577:
            if feature_type == 'patch':
                return image_features[:, 1:, :]
            else:
                return image_features

        # ========= vision backbone without cls =========
        if self.vision_backbone == 'siglip':
            assert image_features.shape[1] == 576 + self.config.num_vision_queries  # TODO: hard code
            return self.feature_select_wo_cls(image_features, feature_type)

        # ========= vision backbone with cls =========
        assert image_features.shape[1] == 577 + self.config.num_vision_queries  # TODO: hard code

        selected_features = None
        if feature_type == 'cls':
            selected_features = image_features[:, 0:1, :]
        elif feature_type == 'query':
            selected_features = image_features[:, -self.config.num_vision_queries:, :]
        elif feature_type == 'patch':
            selected_features = image_features[:, 1:-self.config.num_vision_queries, :]
        elif feature_type == 'mix_cls_patch':
            selected_features = image_features[:, :-self.config.num_vision_queries, :]
        elif feature_type == 'mix_cls_query':
            selected_features = torch.concat([
                image_features[:, 0:1, :], image_features[:, -self.config.num_vision_queries:, :]
            ], dim=1)
        else:
            selected_features = image_features

        return selected_features

    def add_crop_embed(self, vision_queries, patch_positions):
        embed_h = self.crop_embedding_h(patch_positions[:, 0])
        embed_w = self.crop_embedding_w(patch_positions[:, 1])
        pos_embed = (embed_h + embed_w) * 0.5
        pos_embed = einops.repeat(pos_embed, 'N D -> N num_token D', num_token=self.config.num_vision_queries)

        out = vision_queries + pos_embed
        return out

    def forward(self, images, patch_positions, num_images):
        if type(images) is list:
            assert False
            image_features = []
            for image in images:
                vision_queries_input = self.vision_queries.expand(1, -1, -1)
                vision_queries_input = self.add_crop_embed(
                    vision_queries_input,
                    patch_positions
                )

                image_forward_outs = self.vision_tower(
                    image.unsqueeze(0),
                    vision_queries_input,
                    patch_positions=patch_positions,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                image_feature = image_forward_outs["last_hidden_state"]
                image_feature = self.feature_select(image_feature, feature_type=self.config.feature_type)
                image_features.append(image_feature)
        else:
            vision_queries_input = self.vision_queries.expand(images.shape[0], -1, -1)
            vision_queries_input = self.add_crop_embed(
                vision_queries_input,
                patch_positions
            )

            image_forward_outs = self.vision_tower(
                images,
                vision_queries_input,
                patch_positions=patch_positions,
                num_images=num_images,
                output_hidden_states=True,
                output_attentions=True,
            )
            image_features = image_forward_outs["last_hidden_state"]

            # assert image_features.shape[1] == 577 + self.config.num_vision_queries
            image_features = self.feature_select(image_features, feature_type=self.config.feature_type)

        return image_features

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not self.vision_tower_config.enable_interactive_attn:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        from icecream import ic
        from_keyword = 'self_attn'
        new_keyword = 'interact_attn'
        if all(new_keyword not in k for k in state_dict.keys()):
            ic('======================')
            copied_weights = {}
            for name, param in state_dict.items():
                if from_keyword in name:
                    copied_weights[name.replace(from_keyword, new_keyword)] = param.clone()
            ic(copied_weights.keys())
            state_dict.update(copied_weights)

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

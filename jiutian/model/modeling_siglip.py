from typing import Optional, Union, List
import os
from icecream import ic
import math
from functools import partial
import einops

import torch
import transformers

from transformers import PretrainedConfig
from transformers.models.siglip.modeling_siglip import *
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SIGLIP_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # NEW
        self.layer_id = layer_id
        self.num_vision_queries = config.num_vision_queries
        self.enable_interactive_attn = config.enable_interactive_attn

        if config.enable_interactive_attn:
            self.interact_attn = SiglipAttention(config)

    def interactive_attention(self, hidden_states, patch_positions, num_images):
        bs, length, dim = hidden_states.shape

        # num_cropped_images = self.get_num_cropped_images(patch_positions)
        num_cropped_images = num_images
        max_length = max([num * self.num_vision_queries for num in num_cropped_images])

        # concat vision queries of the same image
        vision_queries = hidden_states[:, -self.num_vision_queries:, :]
        vision_queries = torch.split(vision_queries, num_cropped_images, dim=0)

        __vision_queries = torch.zeros(
            (len(num_cropped_images), max_length, dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        attention_mask = torch.ones(
            (len(num_cropped_images), max_length), dtype=torch.long, device=hidden_states.device
        )
        for idx, x in enumerate(vision_queries):
            x_flatten = x.reshape((-1, dim))

            len_padding = max_length - x_flatten.shape[0]

            __vision_queries[idx, :x_flatten.shape[0], :] = x_flatten

            if len_padding > 0:
                attention_mask[idx, -len_padding:] = 0

        vision_queries = __vision_queries  # (n_images, L, D)

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        assert vision_queries.shape == (len(num_cropped_images), max_length, dim)
        assert attention_mask.shape == (len(num_cropped_images), 1, max_length, max_length)

        # attention forward
        query_states, query_attn_weights = self.interact_attn(
            hidden_states=vision_queries,
            attention_mask=attention_mask,
            # causal_attention_mask=None,
            output_attentions=True,
        )

        # split vision queries to original shape
        query_states = torch.concat([
            x[:n_img * self.num_vision_queries, :].reshape((n_img, self.num_vision_queries, -1))
            for n_img, x in zip(num_cropped_images, query_states)
        ], dim=0)
        assert query_states.shape == (bs, self.num_vision_queries, dim)

        # hidden_states[:, -self.num_vision_queries:, :] = query_states
        hidden_states[:, -self.num_vision_queries:, :] = query_states + hidden_states[:, -self.num_vision_queries:, :]

        return hidden_states

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_positions: Optional[torch.Tensor] = None,  # NEW
        num_images: Optional[List] = None,  # NEW
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        if self.enable_interactive_attn:
            hidden_states = self.interactive_attention(hidden_states, patch_positions, num_images)  # NEW

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        patch_positions: Optional[torch.Tensor] = None,  # NEW
        num_images: Optional[List] = None,  # NEW
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    patch_positions=patch_positions,  # NEW
                    num_images=num_images,  # NEW
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    patch_positions=patch_positions,  # NEW
                    num_images=num_images,  # NEW
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


def siglip_vision_transformer_forward(
    self,
    pixel_values,
    vision_queries: Optional[torch.FloatTensor] = None,  # NEW
    patch_positions: Optional[torch.LongTensor] = None,  # NEW
    num_images: Optional[List] = None,  # NEW
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    interpolate_pos_encoding: Optional[bool] = False,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    # ============== add queries ==============
    if vision_queries is not None:
        assert vision_queries.shape == (hidden_states.shape[0], self.config.num_vision_queries, self.config.hidden_size)
        hidden_states = torch.cat([
            hidden_states,
            vision_queries.to(hidden_states.dtype).to(hidden_states.device)
        ], dim=1)
    # =========================================

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        patch_positions=patch_positions,  # NEW
        num_images=num_images,  # NEW
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.post_layernorm(last_hidden_state)

    pooler_output = self.head(last_hidden_state) if self.use_head else None
    if not return_dict:
        return (last_hidden_state, pooler_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def siglip_vision_model_forward(
    self,
    pixel_values,
    vision_queries: Optional[torch.FloatTensor] = None,  # NEW
    patch_positions: Optional[torch.LongTensor] = None,  # NEW
    num_images: Optional[List] = None,  # NEW
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    interpolate_pos_encoding: bool = False,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    Examples:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, SiglipVisionModel

    >>> model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled features
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    return self.vision_model(
        pixel_values=pixel_values,
        vision_queries=vision_queries,  # NEW
        patch_positions=patch_positions,  # NEW
        num_images=num_images,  # NEW
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )


def replace_siglip_forward():
    transformers.models.siglip.modeling_siglip.SiglipEncoder = SiglipEncoder
    transformers.models.siglip.modeling_siglip.SiglipEncoderLayer = SiglipEncoderLayer
    transformers.models.siglip.modeling_siglip.SiglipVisionModel.forward = siglip_vision_model_forward
    transformers.models.siglip.modeling_siglip.SiglipVisionTransformer.forward = siglip_vision_transformer_forward


if __name__ == '__main__':
    replace_siglip_forward()
    print("Test TODO")
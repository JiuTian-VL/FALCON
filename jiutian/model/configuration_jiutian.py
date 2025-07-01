import os
from typing import Union

from transformers import LlamaConfig, PretrainedConfig
from transformers.utils import logging


class JiutianVisionConfig(PretrainedConfig):
    model_type = "jiutian_vision_model"

    def __init__(
        self,
        model_name_or_path=None,
        feature_type='patch',
        num_vision_queries=64,
        anchor_max=9,
        enable_interactive_attn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.feature_type = feature_type
        self.num_vision_queries = num_vision_queries
        self.anchor_max = anchor_max
        self.enable_interactive_attn = enable_interactive_attn


class JiutianProjectorConfig(PretrainedConfig):
    model_type = "jiutian_projector"

    def __init__(
        self,
        projector_type='linear',
        input_dim=1024,
        hidden_size=4096,
        output_dim=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim


class JiutianProcessorConfig(PretrainedConfig):
    def __init__(
        self,
        image_size=336,
        anchors='grid_9',
        add_global_img=True,
        add_textual_crop_indicator=True,
        enable_low_res=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.anchors = anchors
        self.add_global_img = add_global_img
        self.add_textual_crop_indicator = add_textual_crop_indicator
        self.enable_low_res = enable_low_res


class JiutianConfig(LlamaConfig):
    model_type = "jiutian"

    def __init__(
        self,
        mm_load=False,
        vision_delay_load=True,
        add_adapter=False,  # Deprecated
        adapter_bottleneck=32,  # Deprecated
        num_query_tokens=64,
        visual_config=None,
        projector_config=None,
        processor_config=None,
        **kwargs
    ):
        """
            - mm_load: Used to delay the initialization of projector during training,
            avoiding errors of size mismatch when using deepspeed zero3,
            and warnings of uninitialized parameters.
            - vision_delay_load: Used to delay the initialization of vision encoder,
            avoiding the WARNING -- 'parameters are not initialized from the checkpoints'.
        """
        self.visual_config = JiutianVisionConfig().to_dict() if visual_config is None else visual_config
        self.projector_config = JiutianProjectorConfig().to_dict() if projector_config is None else projector_config
        self.processor_config = JiutianProcessorConfig().to_dict() if processor_config is None else processor_config

        self.mm_load = mm_load
        self.vision_delay_load = vision_delay_load
        self.add_adapter = add_adapter  # Deprecated
        self.adapter_bottleneck = adapter_bottleneck  # Deprecated
        self.num_query_tokens = num_query_tokens

        super().__init__(**kwargs)


if __name__ == "__main__":
    print(JiutianVisionConfig().to_dict())
    print(JiutianProjectorConfig().to_dict())
    print(JiutianConfig().to_dict())
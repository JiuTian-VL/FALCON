import math
import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'projector_type', 'linear')

    input_dim = config.input_dim
    output_dim = config.output_dim
    hidden_size = config.hidden_size

    if projector_type == 'linear':
        return nn.Linear(input_dim, output_dim)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))

        if mlp_depth == 1:
            raise ValueError(f"Use 'linear' instead of mlp with depth 1.")

        modules = [nn.Linear(input_dim, hidden_size)]
        for _ in range(1, mlp_depth - 1):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))

        modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_size, output_dim))

        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

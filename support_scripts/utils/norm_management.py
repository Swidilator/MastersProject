import torch
from torch import nn

from typing import Optional


def norm_selector(
    norm_type: str,
    input_height_width: tuple = None,
    output_channel_count: int = None,
    conv: nn.Conv2d = None,
):
    if norm_type == "none":
        out_norm = torch.nn.Identity()
    elif norm_type == "full":
        if output_channel_count is None or output_channel_count is None:
            raise AttributeError
        out_norm = nn.LayerNorm(torch.Size([output_channel_count, *input_height_width]))
    elif norm_type == "half":
        if input_height_width is None:
            raise AttributeError
        out_norm = nn.LayerNorm(torch.Size(input_height_width))
    elif norm_type == "group":
        if input_height_width is None:
            raise AttributeError
        out_norm = nn.GroupNorm(1, output_channel_count)
    elif norm_type == "instance":
        if output_channel_count is None:
            raise AttributeError
        out_norm = nn.InstanceNorm2d(
            output_channel_count,
            eps=1e-05,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
        )
    elif norm_type == "weight":
        if conv is None:
            raise AttributeError
        out_norm = torch.nn.Identity()
        conv = nn.utils.weight_norm(conv)
    else:
        raise ValueError("Incorrect norm type entered.")
    return out_norm, conv

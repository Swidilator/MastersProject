import torch
from torch import nn

from support_scripts.components import RMBlock


class ResNetBlock(nn.Module):
    def __init__(
        self, channel_count: int, input_height_width: tuple, no_add: bool = False
    ):
        """
        A fancy resnext block with grouped convolutions

        """
        super().__init__()

        self.no_add: bool = no_add

        self.rm_block_1 = RMBlock(
            channel_count // 2,
            channel_count,
            input_height_width,
            1,
            "weight",
            1,
        )

        self.rm_block_2 = RMBlock(
            channel_count // 2,
            channel_count // 2,
            input_height_width,
            3,
            "weight",
            32,
        )

        self.rm_block_3 = RMBlock(
            channel_count,
            channel_count // 2,
            input_height_width,
            1,
            "weight",
            1,
        )

    def forward(self, x: torch.Tensor):
        out = self.rm_block_1(x)
        out = self.rm_block_2(out, "before")
        out = self.rm_block_3(out, "before")
        if self.no_add:
            return out
        else:
            return (out + x) / 2

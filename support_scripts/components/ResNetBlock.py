import torch
from torch import nn

from support_scripts.components import RMBlock


class ResNetBlock(nn.Module):
    def __init__(
        self,
        channel_count: int,
        input_height_width: tuple,
    ):
        super().__init__()
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
        return (out + x) / 2

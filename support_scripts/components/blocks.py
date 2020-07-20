import torch
from torch import nn as nn
from torch.nn import modules as modules


class Block(torch.nn.Module):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(Block, self).__init__()
        self.filter_count: int = filter_count
        self.input_channel_count: int = input_channel_count

    @staticmethod
    def init_conv_weights(conv: nn.Module) -> None:
        nn.init.normal_(conv.weight, mean=0.0, std=0.02)
        # nn.init.xavier_uniform_(conv.weight, gain=1)
        # nn.init.constant_(conv.bias, 0)

    @property
    def get_output_filter_count(self) -> int:
        return self.filter_count


class EncoderBlock(Block):
    def __init__(
        self,
        filter_count: int,
        input_channel_count: int,
        kernel_size: int,
        stride: int,
        padding_size: int,
        use_reflect_pad: bool,
        use_instance_norm: bool,
        transpose_conv: bool,
        use_relu: bool,
    ):
        super(EncoderBlock, self).__init__(filter_count, input_channel_count)

        self.padding_size: int = padding_size
        self.use_reflect_pad: bool = use_reflect_pad
        self.transpose_conv: bool = transpose_conv
        self.use_ReLU: bool = use_relu
        self.use_instance_norm: bool = use_instance_norm
        conv_padding_size: int = padding_size

        if self.use_reflect_pad:
            self.reflect_pad: nn.ReflectionPad2d = nn.ReflectionPad2d(padding_size)
            conv_padding_size = 0

        if not self.transpose_conv:
            self.conv_1: modules.Conv2d = nn.Conv2d(
                self.input_channel_count,
                self.filter_count,
                kernel_size=kernel_size,
                stride=stride,
                padding=conv_padding_size,
            )
        else:
            self.conv_1: modules.ConvTranspose2d = nn.ConvTranspose2d(
                self.input_channel_count,
                self.filter_count,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_size,
                output_padding=1,
            )

        Block.init_conv_weights(conv=self.conv_1)

        if self.use_instance_norm:
            self.instance_norm_1 = nn.InstanceNorm2d(
                filter_count,
                eps=1e-05,
                momentum=0.1,
                affine=False,
                track_running_stats=False,
            )

        if self.use_ReLU:
            self.ReLU: nn.ReLU = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x
        if self.use_reflect_pad:
            out = self.reflect_pad(out)
        out = self.conv_1(out)
        if self.use_instance_norm:
            out = self.instance_norm_1(out)
        if self.use_ReLU:
            out = self.ReLU(out)
        return out

import torch
import torch.nn as nn
import torch.nn.modules as modules


class Block(torch.nn.Module):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(Block, self).__init__()
        self.filter_count: int = filter_count
        self.input_channel_count: int = input_channel_count

    @staticmethod
    def init_conv_weights(conv: nn.Module) -> None:
        nn.init.normal_(conv.weight, mean=0.0, std=0.02)
        # nn.init.xavier_uniform_(conv.weight, gain=1)
        nn.init.constant_(conv.bias, 0)

    @property
    def get_output_filter_count(self) -> int:
        return self.filter_count


class ResidualBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(ResidualBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 3
        stride: int = 1
        padding: int = 1

        self.ReLU: nn.ReLU = nn.ReLU(True)

        self.reflect_pad: nn.ReflectionPad2d = nn.ReflectionPad2d(padding)

        self.conv_1: modules.Conv2d = nn.Conv2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        Block.init_conv_weights(conv=self.conv_1)

        # self.conv_2: modules.Conv2d = nn.Conv2d(
        #     self.filter_count,
        #     self.filter_count,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        # )
        # Block.init_conv_weights(conv=self.conv_2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.reflect_pad(input)
        out = self.conv_1(out)
        out = self.ReLU(out)
        # out = self.conv_2(out)
        # out = self.ReLU(out)
        return out + input


class CCIRBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(CCIRBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 7
        stride: int = 1
        padding: int = 3

        self.reflect_pad: nn.ReflectionPad2d = nn.ReflectionPad2d(padding)

        self.ReLU: nn.ReLU = nn.ReLU(True)

        self.conv_1: modules.Conv2d = nn.Conv2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        Block.init_conv_weights(conv=self.conv_1)

        self.instance_norm_1 = nn.InstanceNorm2d(filter_count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.reflect_pad(input)
        out = self.conv_1(out)
        out = self.instance_norm_1(out)
        out = self.ReLU(out)
        return out


class DCIRBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(DCIRBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 3
        stride: int = 2
        padding: int = 1

        self.ReLU: nn.ReLU = nn.ReLU(True)

        self.conv_1: modules.Conv2d = nn.Conv2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        Block.init_conv_weights(conv=self.conv_1)

        self.instance_norm_1 = nn.InstanceNorm2d(filter_count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(input)
        out = self.instance_norm_1(out)
        out = self.ReLU(out)
        return out


class UCIRBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(UCIRBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 3
        stride: float = 2
        padding: int = 1
        output_padding: int = 1

        self.ReLU: nn.ReLU = nn.ReLU(True)

        self.deconv_1: nn.ConvTranspose2d = nn.ConvTranspose2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        Block.init_conv_weights(conv=self.deconv_1)

        self.instance_norm_1 = nn.InstanceNorm2d(filter_count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.deconv_1(input)
        out = self.instance_norm_1(out)
        out = self.ReLU(out)
        return out


class CCILBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int, first_layer: bool):
        super(CCILBlock, self).__init__(filter_count, input_channel_count)

        self.first_layer = first_layer

        kernel_size: int = 4
        stride: int = 2
        padding: int = 0

        self.LReLU: nn.LeakyReLU = nn.LeakyReLU(0.2)
        self.conv_1: modules.Conv2d = nn.Conv2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        Block.init_conv_weights(conv=self.conv_1)

        self.instance_norm_1 = nn.InstanceNorm2d(filter_count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv_1(input)
        if not self.first_layer:
            out = self.instance_norm_1(out)
        out = self.LReLU(out)
        return out
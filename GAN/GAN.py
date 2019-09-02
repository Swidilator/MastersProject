import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
from math import log2
import time
import random

from Helper_Stuff import *
from Data_Management import GANDataset
from CRN.Perceptual_Loss import PerceptualLossNetwork
from Training_Framework import MastersModel

import wandb





class Generator(torch.nn.Module):
    def __init__(self, input_channel_count: int):
        super(Generator, self).__init__()
        self.input_channel_count = input_channel_count

        self.global_generator: GlobalGenerator = GlobalGenerator(input_channel_count)
        self.local_enhancer: LocalEnhancer = LocalEnhancer(input_channel_count)

    def forward(self, inputs: gan_input) -> torch.Tensor:
        # TODO Check that this is dividing properly
        half_size: tuple = (int(inputs[0].shape[2] / 2), int(inputs[0].shape[3] / 2))

        smaller_input = torch.nn.functional.interpolate(
            input=inputs[0],
            size=half_size,
            mode="nearest",
        )

        global_output: torch.Tensor = self.global_generator(smaller_input)
        local_output: torch.Tensor = self.local_enhancer((inputs[0], global_output))

        return local_output


class Block(torch.nn.Module):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(Block, self).__init__()
        self.filter_count: int = filter_count
        self.input_channel_count: int = input_channel_count

    @staticmethod
    def init_conv_weights(conv: nn.Module) -> None:
        nn.init.xavier_uniform_(conv.weight, gain=1)
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

        self.ReLU: nn.ReLU = nn.ReLU()

        self.conv_1: modules.Conv2d = nn.Conv2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        Block.init_conv_weights(conv=self.conv_1)

        self.conv_2: modules.Conv2d = nn.Conv2d(
            self.filter_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        Block.init_conv_weights(conv=self.conv_2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv_1(input)
        out = self.ReLU(out)
        out = self.conv_2(out)
        out = self.ReLU(out)
        return out + input


class CCIRBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(CCIRBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 7
        stride: int = 1
        padding: int = 3

        self.ReLU: nn.ReLU = nn.ReLU()
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
        out = self.instance_norm_1(out)
        out = self.ReLU(out)
        return out


class DCIRBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(DCIRBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 3
        stride: int = 2
        padding: int = 0

        self.ReLU: nn.ReLU = nn.ReLU()

        self.reflect_pad: nn.ReflectionPad2d = nn.ReflectionPad2d(1)

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
        out: torch.Tensor = self.reflect_pad(input)
        out = self.conv_1(out)
        out = self.instance_norm_1(out)
        out = self.ReLU(out)
        return out


class UCIRBlock(Block):
    def __init__(self, filter_count: int, input_channel_count: int):
        super(UCIRBlock, self).__init__(filter_count, input_channel_count)

        kernel_size: int = 3
        stride: float = 2
        padding: int = 1

        self.ReLU: nn.ReLU = nn.ReLU()

        self.deconv_1: nn.ConvTranspose2d = nn.ConvTranspose2d(
            self.input_channel_count,
            self.filter_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=1
        )
        Block.init_conv_weights(conv=self.deconv_1)

        self.instance_norm_1 = nn.InstanceNorm2d(filter_count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.deconv_1(input)
        out = self.instance_norm_1(out)
        out = self.ReLU(out)
        return out


class GlobalGenerator(torch.nn.Module):
    def __init__(self, input_channel_count: int) -> None:
        super(GlobalGenerator, self).__init__()

        # TODO Convert to list
        self.c1: CCIRBlock = CCIRBlock(64, input_channel_count)

        self.d2: DCIRBlock = DCIRBlock(128, self.c1.get_output_filter_count)
        self.d3: DCIRBlock = DCIRBlock(256, self.d2.get_output_filter_count)
        self.d4: DCIRBlock = DCIRBlock(512, self.d3.get_output_filter_count)
        self.d5: DCIRBlock = DCIRBlock(1024, self.d4.get_output_filter_count)

        self.r6: ResidualBlock = ResidualBlock(1024, self.d5.get_output_filter_count)
        self.r7: ResidualBlock = ResidualBlock(1024, self.r6.get_output_filter_count)
        self.r8: ResidualBlock = ResidualBlock(1024, self.r7.get_output_filter_count)
        self.r9: ResidualBlock = ResidualBlock(1024, self.r8.get_output_filter_count)
        self.r10: ResidualBlock = ResidualBlock(1024, self.r9.get_output_filter_count)
        self.r11: ResidualBlock = ResidualBlock(1024, self.r10.get_output_filter_count)
        self.r12: ResidualBlock = ResidualBlock(1024, self.r11.get_output_filter_count)
        self.r13: ResidualBlock = ResidualBlock(1024, self.r12.get_output_filter_count)
        self.r14: ResidualBlock = ResidualBlock(1024, self.r13.get_output_filter_count)

        self.u15: UCIRBlock = UCIRBlock(512, self.r14.get_output_filter_count)
        self.u16: UCIRBlock = UCIRBlock(256, self.u15.get_output_filter_count)
        self.u17: UCIRBlock = UCIRBlock(128, self.u16.get_output_filter_count)
        self.u18: UCIRBlock = UCIRBlock(64, self.u17.get_output_filter_count)

        self.c19: CCIRBlock = CCIRBlock(3, self.u18.get_output_filter_count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.c1(input)

        out = self.d2(out)
        out = self.d3(out)
        out = self.d4(out)
        out = self.d5(out)

        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)
        out = self.r11(out)
        out = self.r12(out)
        out = self.r13(out)
        out = self.r14(out)

        out = self.u15(out)
        out = self.u16(out)
        out = self.u17(out)
        out = self.u18(out)

        # out = self.c19(out)
        return out


class LocalEnhancer(torch.nn.Module):
    def __init__(self, input_channel_count):
        super(LocalEnhancer, self).__init__()

        self.c1: CCIRBlock = CCIRBlock(64, input_channel_count)

        self.d2: DCIRBlock = DCIRBlock(64, self.c1.get_output_filter_count)

        self.r3: ResidualBlock = ResidualBlock(64, self.d2.get_output_filter_count)
        self.r4: ResidualBlock = ResidualBlock(64, self.r3.get_output_filter_count)
        self.r5: ResidualBlock = ResidualBlock(64, self.r4.get_output_filter_count)

        self.u6: UCIRBlock = UCIRBlock(32, self.r5.get_output_filter_count)

        self.u7: UCIRBlock = UCIRBlock(3, self.u6.get_output_filter_count)

    def forward(self, inputs: generator_input) -> torch.Tensor:
        out: torch.Tensor = self.c1(inputs[0])

        # Add the output of the global generator
        out = self.d2(out) + inputs[1]

        out = self.r3(out)
        out = self.r4(out)
        out = self.r5(out)

        out = self.u6(out)
        out = self.u7(out)

        return out

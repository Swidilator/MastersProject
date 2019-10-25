import torch
import torch.nn as nn
import torch.nn.modules as modules

from typing import Tuple, List, Any
# from math import log2
# import time
# import random

from GAN.Blocks import *

import wandb


class Generator(torch.nn.Module):
    def __init__(self, input_channel_count: int):
        super(Generator, self).__init__()
        self.input_channel_count = input_channel_count

        self.global_generator: GlobalGenerator = GlobalGenerator(input_channel_count)
        self.local_enhancer: LocalEnhancer = LocalEnhancer(input_channel_count)

    def forward(self, inputs: Tuple[torch.Tensor, bool]) -> torch.Tensor:
        # TODO Check that this is dividing properly
        half_size: tuple = (inputs[0].shape[2] // 2, inputs[0].shape[3] // 2)

        smaller_input = torch.nn.functional.interpolate(
            input=inputs[0], size=half_size, mode="nearest"
        )

        global_output: torch.Tensor = self.global_generator(smaller_input)
        local_output: torch.Tensor = self.local_enhancer((inputs[0], global_output))

        return local_output


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

        img = self.c19(out)
        return out, img


class LocalEnhancer(torch.nn.Module):
    def __init__(self, input_channel_count):
        super(LocalEnhancer, self).__init__()
        self.reflect_pad: nn.ReflectionPad2d = nn.ReflectionPad2d(3)
        self.c1: CCIRBlock = CCIRBlock(64, input_channel_count)

        self.d2: DCIRBlock = DCIRBlock(64, self.c1.get_output_filter_count)

        self.r3: ResidualBlock = ResidualBlock(64, self.d2.get_output_filter_count)
        self.r4: ResidualBlock = ResidualBlock(64, self.r3.get_output_filter_count)
        self.r5: ResidualBlock = ResidualBlock(64, self.r4.get_output_filter_count)

        self.u6: UCIRBlock = UCIRBlock(32, self.r5.get_output_filter_count)

        self.c7: CCIRBlock = CCIRBlock(3, self.u6.get_output_filter_count)

        self.tan_h = nn.Tanh()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # out: torch.Tensor = self.reflect_pad(inputs[0])
        out: torch.Tensor = self.c1(inputs[0])
        out = self.tan_h(out).clone()

        # Add the output of the global generator

        out = self.d2(out) + inputs[1][0]
        # del inputs

        out = self.r3(out)
        out = self.r4(out)
        out = self.r5(out)

        out = self.u6(out)
        out = self.c7(out)

        out = self.tan_h(out).clone()
        return out, inputs[1][1]

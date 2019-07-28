import torch
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
import torchvision.transforms as transforms
from PIL import Image
from math import log2


class RefinementModule(modules.Module):
    r"""
    One 3 layer module making up a segment of a CRN. Mask input tensor & prior layers get resized.

    Args:
        prior_layer_channel_count(int): number of input channels from previous layer
        semantic_input_channel_count(int): number of input channels from semantic annotation
        output_channel_count(int): number of output channels
        input_height_width(tuple(int)): input image height and width
        is_final_module(bool): is this the final module in the network
    """

    def __init__(
        self,
        prior_layer_channel_count: int,
        semantic_input_channel_count: int,
        output_channel_count: int,
        input_height_width: tuple,
        is_final_module: bool = False,
    ):
        super(RefinementModule, self).__init__()

        self.input_height_width: tuple = input_height_width
        self.total_input_channel_count: int = (
            prior_layer_channel_count + semantic_input_channel_count
        )
        self.output_channel_count: int = output_channel_count
        self.is_final_module: bool = False

        # Module architecture
        self.conv_1 = nn.Conv2d(
            self.total_input_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer_norm_1 = nn.LayerNorm(
            RefinementModule.change_output_channel_size(
                input_height_width, self.output_channel_count
            )
        )

        self.conv_2 = nn.Conv2d(
            self.output_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer_norm_2 = nn.LayerNorm(
            RefinementModule.change_output_channel_size(
                input_height_width, self.output_channel_count
            )
        )

        self.conv_3 = nn.Conv2d(
            self.output_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if not is_final_module:
            self.layer_norm_3 = nn.LayerNorm(
                RefinementModule.change_output_channel_size(
                    input_height_width, self.output_channel_count
                )
            )

        self.leakyReLU = nn.LeakyReLU()

    @staticmethod
    def change_output_channel_size(
        input_height_width: tuple, output_channel_number: int
    ):
        size_list = list(input_height_width)
        size_list.insert(0, output_channel_number)
        # print(size_list)
        return torch.Size(size_list)

    def forward(self, inputs: list):
        mask: torch.Tensor = inputs[0]
        prior_layers: torch.Tensor = inputs[1]
        mask = torch.nn.functional.interpolate(
            input=mask, size=self.input_height_width, mode="nearest"
        )

        prior_layers = torch.nn.functional.interpolate(
            input=prior_layers, size=self.input_height_width, mode="bilinear"
        )

        x = torch.cat((mask, prior_layers), dim=1)

        x = self.conv_1(x)
        # print(x.size())
        x = self.layer_norm_1(x)
        x = self.leakyReLU(x)

        x = self.conv_2(x)
        # print(x.size())
        x = self.layer_norm_2(x)
        x = self.leakyReLU(x)

        x = self.conv_3(x)
        # print(x.size())
        if not self.is_final_module:
            x = self.layer_norm_1(x)
            x = self.leakyReLU(x)
        return x


# TODO Fill with actual code, currently old network
class CRN(torch.nn.Module):
    def __init__(
        self,
        input_tensor_size: tuple,
        final_image_size: tuple,
        num_output_images: int,
        num_classes: int,
    ):
        super(CRN, self).__init__()

        self.input_tensor_size: tuple = input_tensor_size
        self.final_image_size: tuple = final_image_size
        self.num_output_images: int = num_output_images
        self.num_classes: int = num_classes

        self.__NUM_NOISE_CHANNELS__: int = 1
        self.__NUM_OUTPUT_IMAGE_CHANNELS__: int = 3

        self.num_rms: int = int(log2(final_image_size[0])) - 1

        self.rms_list: list = []

        self.rms_list.append(
            RefinementModule(
                prior_layer_channel_count=self.__NUM_NOISE_CHANNELS__,
                semantic_input_channel_count=num_classes,
                output_channel_count=1024,
                input_height_width=input_tensor_size,
                is_final_module=False,
            )
        )

        for i in range(1, self.num_rms - 1):
            self.rms_list.append(
                RefinementModule(
                    prior_layer_channel_count=1024,
                    semantic_input_channel_count=num_classes,
                    output_channel_count=1024,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    is_final_module=False,
                )
            )

        self.rms_list.append(
            RefinementModule(
                prior_layer_channel_count=1024,
                semantic_input_channel_count=num_classes,
                output_channel_count=self.__NUM_OUTPUT_IMAGE_CHANNELS__
                * num_output_images,
                input_height_width=final_image_size,
                is_final_module=True,
            )
        )

        self.rms = nn.Sequential(*self.rms_list)

    def __del__(self):
        del self.rms

    def forward(self, inputs: list):
        mask: torch.Tensor = inputs[0]
        noise: torch.Tensor = inputs[1]
        batch_size: int = inputs[2]
        x: torch.Tensor = self.rms[0]([mask, noise])
        for i in range(1, self.num_rms):
            x = self.rms[i]([mask, x])
        return x

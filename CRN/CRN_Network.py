import torch
import torch.nn as nn

from math import log2

from Helper_Stuff import *
from CRN.Refinement_Module import RefinementModule


class CRN(torch.nn.Module):
    def __init__(
        self,
        input_tensor_size: image_size,
        final_image_size: image_size,
        num_output_images: int,
        num_classes: int,
        num_inner_channels: int,
    ):
        super(CRN, self).__init__()

        self.input_tensor_size: image_size = input_tensor_size
        self.final_image_size: image_size = final_image_size
        self.num_output_images: int = num_output_images
        self.num_classes: int = num_classes
        self.num_inner_channels: int = num_inner_channels

        self.__NUM_NOISE_CHANNELS__: int = 0
        self.__NUM_OUTPUT_IMAGE_CHANNELS__: int = 3

        self.num_rms: int = int(log2(final_image_size[0])) - 1

        self.rms_list: nn.ModuleList = nn.ModuleList(
            [
                RefinementModule(
                    prior_layer_channel_count=self.__NUM_NOISE_CHANNELS__,
                    semantic_input_channel_count=num_classes,
                    output_channel_count=self.num_inner_channels,
                    input_height_width=input_tensor_size,
                    is_final_module=False,
                )
            ]
        )

        self.rms_list.extend(
            [
                RefinementModule(
                    prior_layer_channel_count=self.num_inner_channels,
                    semantic_input_channel_count=num_classes,
                    output_channel_count=self.num_inner_channels,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    is_final_module=False,
                )
                for i in range(1, self.num_rms - 1)
            ]
        )

        self.rms_list.append(
            RefinementModule(
                prior_layer_channel_count=self.num_inner_channels,
                semantic_input_channel_count=num_classes,
                output_channel_count=self.num_inner_channels,
                input_height_width=final_image_size,
                is_final_module=True,
                final_channel_count=self.__NUM_OUTPUT_IMAGE_CHANNELS__
                * num_output_images,
            )
        )

        self.tan_h = nn.Tanh()

    def forward(self, inputs: list):
        mask: torch.Tensor = inputs[0]
        noise: torch.Tensor = inputs[1]

        x: torch.Tensor = self.rms_list[0]([mask, noise])
        for i in range(1, len(self.rms_list)):
            x = self.rms_list[i]([mask, x])
        # TanH for squeezing outputs to [-1, 1]
        x = self.tan_h(x).clone()
        return x
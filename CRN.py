import torch
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
import torchvision.transforms as transforms
from PIL import Image
from math import log2


class RefinementModule(modules.Module):
    r"""
    One 3 layer module making up a segment of a CRN. Mask input tensor is of correct size, prior layers get resized.

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

        # self.input_resize_transform = transforms.Compose(
        #     [
        #         transforms.Resize(
        #             self.input_height_width,
        #             Image.NEAREST,  # NEAREST as the values are categories and are not continuous
        #         ),
        #         transforms.ToTensor(),
        #     ]
        # )
        #
        self.prior_layer_resize_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    self.input_height_width,
                    Image.NEAREST,  # NEAREST as the values are categories and are not continuous
                ),
                transforms.ToTensor(),
            ]
        )
        #
        # self.mask_resize_transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(
        #             self.input_height_width,
        #             Image.NEAREST,  # NEAREST as the values are categories and are not continuous
        #         ),
        #         transforms.ToTensor(),
        #         transforms.Lambda(lambda x: (x * 255).long()[0]),
        #         transforms.Lambda(lambda x: one_hot(x, num_classes)),
        #     ]
        # )

        # Module architecture
        self.conv_1 = nn.Conv2d(
            self.total_input_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer_norm_1 = nn.LayerNorm(
            change_output_channel_size(input_height_width, self.output_channel_count)
        )

        self.conv_2 = nn.Conv2d(
            self.output_channel_count,
            self.output_channel_count,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layer_norm_2 = nn.LayerNorm(
            change_output_channel_size(input_height_width, self.output_channel_count)
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
                change_output_channel_size(
                    input_height_width, self.output_channel_count
                )
            )

        self.leakyReLU = nn.LeakyReLU()

    def forward(self, mask: torch.Tensor, prior_layers: torch.Tensor):
        if prior_layers is not None:
            # prior_layers = self.prior_layer_resize_transform(prior_layers)
            prior_layers2 = torch.stack([
                self.prior_layer_resize_transform(x_i) for i, x_i in enumerate(torch.unbind(prior_layers, dim=1))
            ], dim=1)
            x = torch.cat((mask, prior_layers2), dim=1)
        else:
            x = mask
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
    def __init__(self, final_image_size: tuple, num_output_images: int = 1, num_classes: int = 35):
        super(CRN, self).__init__()

        self.num_output_images = num_output_images
        self.final_image_size = final_image_size
        self.num_classes = num_classes

        self.num_rms: int = int(log2(final_image_size[0])) - 1

        self.rms: list = []

        self.rms.append(
            RefinementModule(
                prior_layer_channel_count=1,
                semantic_input_channel_count=num_classes,
                output_channel_count=1024,
                input_height_width=(4, 8),
                is_final_module=False
            )
        )

        for i in range(1, self.num_rms - 1):
            self.rms.append(
                RefinementModule(
                    prior_layer_channel_count=1024,
                    semantic_input_channel_count=num_classes,
                    output_channel_count=1024,
                    input_height_width=(2 ** (i + 2), 2 ** (i + 3)),
                    is_final_module=False
                )
            )

        self.rms.append(
            RefinementModule(
                prior_layer_channel_count=1024,
                semantic_input_channel_count=num_classes,
                output_channel_count=3 * num_output_images,
                input_height_width=final_image_size,
                is_final_module=True
            )
        )

    def forward(self, masks: torch.Tensor):
        sizes: tuple = self.rms[0].input_height_width
        mask_view: torch.Tensor = masks[:, 0:self.num_classes, 0:sizes[0], 0:sizes[1]]
        mask_view_shape = mask_view.shape
        # print(masks[:, 0:self.num_classes, 0:sizes[0], 0:sizes[1]].shape)
        noise: torch.Tensor = torch.randn(mask_view_shape[2], mask_view_shape[3]).unsqueeze(0).unsqueeze(0)
        x: torch.Tensor = self.rms[0](mask_view, noise)

        for i in range(1, self.num_rms):
            sizes: tuple = self.rms[i].input_height_width
            mask_view = masks[:, i * self.num_classes: (i + 1) * self.num_classes, 0:sizes[0], 0:sizes[1]]
            x = self.rms[i](
                mask_view, x
            )
        return x


def change_output_channel_size(input_height_width: tuple, output_channel_number: int):
    size_list = list(input_height_width)
    size_list.insert(0, output_channel_number)
    # print(size_list)
    return torch.Size(size_list)

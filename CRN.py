import torch
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
from math import log2

from Data_Types import image_size, epoch_output
from Data_Management import CRNDataset
from Perceptual_Loss import PerceptualLossNetwork
from Training_Framework import MastersModel


class CRNFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        input_tensor_size: image_size,
        max_input_height_width: image_size,
        num_output_images: int,
        num_classes: int,
        batch_size: int,
        num_loader_workers: int,
    ):
        super(CRNFramework, self).__init__(device=device)
        self.batch_size = batch_size
        self.input_tensor_size = input_tensor_size

        self.__set_data_loader__(
            data_path,
            max_input_height_width,
            num_classes,
            batch_size,
            num_loader_workers,
        )
        self.__set_model__(
            input_tensor_size, max_input_height_width, num_output_images, num_classes
        )

    def __set_data_loader__(
        self,
        data_path,
        max_input_height_width,
        num_classes,
        batch_size,
        num_loader_workers,
    ):
        self.__data_set__ = CRNDataset(
            max_input_height_width=max_input_height_width,
            root="../CityScapes Samples/Train/",
            split="train",
            num_classes=num_classes,
        )

        self.data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

    def __set_model__(
        self, input_tensor_size, max_input_height_width, num_output_images, num_classes
    ) -> None:
        self.crn: CRN = CRN(
            input_tensor_size=input_tensor_size,
            final_image_size=max_input_height_width,
            num_output_images=num_output_images,
            num_classes=num_classes,
        )
        self.crn = self.crn.to(self.device)

        self.optimizer = torch.optim.SGD(self.crn.parameters(), lr=0.01, momentum=0.9)
        self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork()
        self.loss_net = self.loss_net.to(self.device)

    def save_model(self, model_dir: str) -> None:
        pass

    def load_model(self, model_dir: str, model_name: str) -> None:
        pass

    def train(self) -> epoch_output:
        self.crn.train()
        torch.cuda.empty_cache()
        loss_sum: float = 0.0
        for batch_idx, (img, msk) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            img: torch.Tensor = img.to(self.device)
            msk: torch.Tensor = msk.to(self.device)
            noise: torch.Tensor = torch.randn(
                self.batch_size,
                1,
                self.input_tensor_size[0],
                self.input_tensor_size[1],
                device=self.device,
            )
            noise = noise.to(self.device)

            out = self.crn(inputs=[msk, noise, self.batch_size])

            out = CRNFramework.__normalise__(out)

            loss: torch.Tensor = self.loss_net([out, img])
            loss.backward()
            loss_sum += loss.item()
            self.optimizer.step()
            del loss, msk, noise, img
        return loss_sum, None

    def eval(self) -> epoch_output:
        pass

    @staticmethod
    def __single_channel_normalise__(
        channel: torch.Tensor, params: tuple
    ) -> torch.Tensor:
        # channel = [H ,W]   params = (mean, std)
        return (channel - params[0]) / params[1]

    @staticmethod
    def __single_image_normalise__(image: torch.Tensor, mean, std) -> torch.Tensor:
        for i in range(3):
            image[i] = CRNFramework.__single_channel_normalise__(
                image[i], (mean[i], std[i])
            )
        return image

    @staticmethod
    def __normalise__(input: torch.Tensor) -> torch.Tensor:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if len(input.shape) == 4:
            for i in range(input.shape[0]):
                input[i] = CRNFramework.__single_image_normalise__(input[i], mean, std)
        else:
            input = CRNFramework.__single_image_normalise__(input, mean, std)
        return input


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
        input_tensor_size: image_size,
        final_image_size: image_size,
        num_output_images: int,
        num_classes: int,
    ):
        super(CRN, self).__init__()

        self.input_tensor_size: image_size = input_tensor_size
        self.final_image_size: image_size = final_image_size
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

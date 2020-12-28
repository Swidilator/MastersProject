import torch
from torch import nn
from typing import Tuple, List

from support_scripts.components.blocks import CCILBlock

Discriminator_Output = Tuple[torch.Tensor, List[torch.Tensor]]


class FullDiscriminator(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_channel_count: int,
        num_discriminators: int,
        use_sigmoid_discriminator: bool,
    ):
        super(FullDiscriminator, self).__init__()

        self.device: torch.device = device

        self.num_discriminators: int = num_discriminators

        self.discriminators: torch.nn.ModuleList = nn.ModuleList()

        for i in range(num_discriminators):
            disc: SingleDiscriminator = SingleDiscriminator(
                self.device, input_channel_count, use_sigmoid_discriminator
            )
            self.discriminators.append(disc.to(self.device))

        self.downsample: nn.AvgPool2d = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def calculate_loss(self, input_tensor: torch.Tensor, label: float, criterion):
        label_tensor = torch.full(
            (input_tensor.shape[0],),
            label,
            device=self.device,
            requires_grad=False,
        )
        result: torch.Tensor = criterion(input_tensor, label_tensor)
        return result

    def forward(
        self, input_tuple: Tuple[torch.Tensor]
    ) -> Discriminator_Output:

        filtered_input_list: list = [x for x in input_tuple if x is not None]

        # Concatenate input tensors together
        input_concat: torch.Tensor = torch.cat(filtered_input_list, dim=1)

        output: torch.Tensor = torch.tensor([], device=self.device)
        # output_split: Optional[List[Feature_Extractions]] = []
        output_feature_extractions: List[torch.Tensor] = []

        for i in reversed(range(self.num_discriminators)):
            if i < self.num_discriminators - 1:
                input_concat: torch.Tensor = self.downsample(input_concat)

            single_output: torch.Tensor
            single_feature_extractions: List[torch.Tensor]
            single_output, single_feature_extractions = self.discriminators[i](
                input_concat
            )

            if output is not None:
                output = torch.cat((output, single_output.view(-1)))
            else:
                output = single_output.view(-1)

            output_feature_extractions += single_feature_extractions

        return output, output_feature_extractions


class SingleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_channel_count: int,
        use_sigmoid_discriminator: bool,
    ):
        super(SingleDiscriminator, self).__init__()

        self.device = device
        self.use_sigmoid_discriminator = use_sigmoid_discriminator

        self.cl1: CCILBlock = CCILBlock(64, input_channel_count, first_layer=True)
        self.cl2: CCILBlock = CCILBlock(128, self.cl1.get_output_filter_count, False)
        self.cl3: CCILBlock = CCILBlock(256, self.cl2.get_output_filter_count, False)
        self.cl4: CCILBlock = CCILBlock(512, self.cl3.get_output_filter_count, False)

        self.final_conv: nn.Conv2d = nn.Conv2d(
            self.cl4.get_output_filter_count, out_channels=1, kernel_size=4, stride=1, padding=2
        )
        if self.use_sigmoid_discriminator:
            self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(
        self, concat_input: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out1 = self.cl1(concat_input)
        out2 = self.cl2(out1)
        out3 = self.cl3(out2)
        out4 = self.cl4(out3)
        out = self.final_conv(out4)
        if self.use_sigmoid_discriminator:
            out = self.sigmoid(out)
        # out = torch.mean(out, axis=(1, 2, 3))

        return out, [out1, out2, out3, out4, concat_input]
        # return out, ((out1, n1), (out2, n2), (out3, n3), (out4, n4))


def feature_matching_error(
    output_extra_real: List[torch.Tensor],
    output_extra_fake: List[torch.Tensor],
    feature_matching_weight: float,
    num_discriminators: int,
) -> torch.Tensor:
    """
    Compute the L1 feature matching error between real and fake image network features.

    :param num_discriminators:
    :param output_extra_real: List of network outputs generated from real image.
    :param output_extra_fake: List of network outputs generated from fake image.
    :param feature_matching_weight: Float for scaling feature matching loss.
    :return: Total error for all features.
    """

    device = output_extra_real[0].device

    total_loss: torch.Tensor = torch.tensor([0.0], device=device)
    total_loss.requires_grad = False

    # Number of feature comparisons to perform
    num_comparisons: int = len(output_extra_fake)
    D_weights = 1.0 / num_discriminators
    feat_weights = 4.0 / num_comparisons

    for i in range(num_comparisons):
        total_loss += (
            torch.nn.functional.l1_loss(
                output_extra_real[i].detach(), output_extra_fake[i]
            )
            * D_weights
            * feat_weights
            * feature_matching_weight
        )

    return total_loss

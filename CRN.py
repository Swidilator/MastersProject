import torch
import torch.nn as nn
import torch.nn.modules as modules
import numpy


class RefinementModule(modules.Module):
    r"""
    One 3 layer module making up a segment of a CRN
    """
    def __init__(
        self,
        input_channel_number: int,
        output_channel_number: int,
        semantic_layer_channel_count: int,
        input_size: torch.Size,
        is_final_module: bool = False,
    ):
        super(RefinementModule, self).__init__()

        self.input_size: torch.Size = input_size
        self.input_number: int = (
            input_channel_number + semantic_layer_channel_count
        )
        self.output_channel_number: int = output_channel_number
        self.is_final_module: bool = False

        self.conv_1 = nn.Conv2d(
            self.input_number, self.output_channel_number, kernel_size=3, stride=1, padding=1
        )
        self.layer_norm_1 = nn.LayerNorm(
            change_output_channel_size(input_size, self.output_channel_number)
        )

        self.conv_2 = nn.Conv2d(
            self.output_channel_number, self.output_channel_number, kernel_size=3, stride=1, padding=1
        )
        self.layer_norm_2 = nn.LayerNorm(
            change_output_channel_size(input_size, self.output_channel_number)
        )

        self.conv_3 = nn.Conv2d(
            self.output_channel_number, self.output_channel_number, kernel_size=3, stride=1, padding=1
        )
        if not is_final_module:
            self.layer_norm_3 = nn.LayerNorm(
                change_output_channel_size(input_size, self.output_channel_number)
            )

        self.leakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_1(x)
        print(x.size())
        x = self.layer_norm_1(x)
        x = self.leakyReLU(x)

        x = self.conv_2(x)
        print(x.size())
        x = self.layer_norm_2(x)
        x = self.leakyReLU(x)

        x = self.conv_3(x)
        print(x.size())
        if not self.is_final_module:
            x = self.layer_norm_1(x)
            x = self.leakyReLU(x)
        return x


class CRN(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(CRN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 0
            torch.nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 6
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # 6
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def change_output_channel_size(input_size: torch.Size, output_channel_number: int):
    size_list = list(input_size[2:])
    size_list.insert(0, output_channel_number)
    print(size_list)
    return torch.Size(size_list)

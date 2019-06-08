import torch
import torch.nn as nn
import torch.nn.modules as modules
import numpy


class RefinementModule(modules.Module):
    def __init__(
        self, parent_module, output_number, semantic_layer_number, w_res=0, h_res=0
    ):
        super(RefinementModule, self).__init__()

        self.w_res = w_res
        self.h_res = h_res
        self.input_number = parent_module.output_number + semantic_layer_number
        self.output_number = output_number

        if self.parent_module is not None:
            self.w_res = parent_module.w_res * 2
            self.h_res = parent_module.h_res * 2

        self.conv_1 = nn.Conv2d(
            self.input_number, self.output_number, kernel_size=3, stride=1, padding=1
        )
        ############
        self.layer_norm_1 = nn.LayerNorm(input.size()[1:])
        ############
        self.conv_2 = tnn.Conv2d(
            self.output_number, self.output_number, kernel_size=3, stride=1, padding=1
        )
        self.conv_3 = nn.Conv2d(
            self.output_number, self.output_number, kernel_size=3, stride=1, padding=1
        )

    def forward(self, input):
        return sparseLinear(input, self.weight, self.bias)


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

import torch
import torch.nn as nn
import torch.nn.modules as modules
import torchvision
from torchvision.transforms import Resize
from copy import copy


def get_layer_values(self: torch.nn.modules.conv.Conv2d, input: tuple, output: torch.Tensor) -> None:
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    self.stored_output: torch.Tensor = output


class PerceptualDifference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        result = (a - b).abs().sum()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PerceptualLossNetwork(modules.Module):
    def __init__(self):
        super(PerceptualLossNetwork, self).__init__()

        self.vgg: torch.nn.Module = torchvision.models.vgg19(pretrained=True, progress=True)
        self.vgg.eval()
        del self.vgg.classifier, self.vgg.avgpool
        for i in self.vgg.features:
            i.requires_grad = False

        self.loss_layer_numbers: tuple = (2, 7, 12, 21, 30)

        for i, num in enumerate(self.loss_layer_numbers):
            self.vgg.features[num].register_forward_hook(get_layer_values)
            self.vgg.features[num].requires_grad = False

        # self.vgg_1 = copy(vgg.features[2])  # 64
        # self.vgg_1.requires_grad = False
        # self.vgg_2 = copy(vgg.features[7])  # 128
        # self.vgg_2.requires_grad = False
        # self.vgg_3 = copy(vgg.features[12])  # 256
        # self.vgg_3.requires_grad = False
        # self.vgg_4 = copy(vgg.features[21])  # 512
        # self.vgg_4.requires_grad = False
        # self.vgg_5 = copy(vgg.features[30])  # 512
        # self.vgg_5.requires_grad = False
        # del vgg

    def forward(self, input, truth):
        result_1 = PerceptualDifference(self.vgg_1(input), self.vgg_1(truth))
        result_2 = PerceptualDifference(self.vgg_2(input), self.vgg_2(truth))
        result_3 = PerceptualDifference(self.vgg_3(input), self.vgg_3(truth))
        result_4 = PerceptualDifference(self.vgg_4(input), self.vgg_4(truth))
        result_5 = PerceptualDifference(self.vgg_5(input), self.vgg_5(truth))
        return result_1, result_2, result_3, result_4, result_5


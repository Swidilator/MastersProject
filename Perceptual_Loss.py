import torch
import torch.nn as nn
import torch.nn.modules as modules
import torchvision
from torchvision.transforms import Resize
from copy import copy


# class PerceptualDifference(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, img, trth):
#         result = (img - trth).abs().sum()
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None


def get_layer_values(
    self: torch.nn.modules.conv.Conv2d, input: tuple, output: torch.Tensor
) -> None:
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    self.stored_output: torch.Tensor = output


class PerceptualLossNetwork(modules.Module):
    def __init__(self):
        super(PerceptualLossNetwork, self).__init__()

        self.vgg: torch.nn.Module = torchvision.models.vgg19(
            pretrained=True, progress=True
        )
        self.vgg.eval()
        del self.vgg.classifier, self.vgg.avgpool
        for i in self.vgg.features:
            i.requires_grad = False

        self.loss_layer_numbers: tuple = (2, 7, 12, 21, 30)

        self.loss_layer_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0]

        for i, num in enumerate(self.loss_layer_numbers):
            self.vgg.features[num].register_forward_hook(get_layer_values)
            self.loss_layer_coefficients[i] = (
                1.0 / self.vgg.features[num].weight.data.numel()
            )

        print("loss_coefficients:", self.loss_layer_coefficients)
        self.norm = torch.nn.modules.normalization

    def forward(self, inputs: list):
        input: torch.Tensor = inputs[0]
        truth: torch.Tensor = inputs[1]
        self.vgg.features(input)
        result_gen: list = []
        for i, num in enumerate(self.loss_layer_numbers):
            result_gen.append(self.vgg.features[num].stored_output)

        self.vgg.features(truth)
        result_truth: list = []
        for i, num in enumerate(self.loss_layer_numbers):
            result_truth.append(self.vgg.features[num].stored_output)

        device = self.vgg.features[0].weight.device
        total_loss = torch.zeros(1).float().to(device)

        # perceptual_difference = PerceptualDifference.apply

        # TODO Implement correct loss
        for i in range(len(result_gen)):
            res = (
                result_truth[i] - result_gen[i]
            ).norm() * self.loss_layer_coefficients[i]
            total_loss += res

        del result_gen, result_truth
        return total_loss

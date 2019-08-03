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
    def __init__(self, input_image_size: tuple):
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

        num_elements_input = 1
        for val in input_image_size:
            num_elements_input *= val

        self.loss_direct_coefficient: float = 1.0/num_elements_input

        for i, num in enumerate(self.loss_layer_numbers):
            self.vgg.features[num].register_forward_hook(get_layer_values)
            self.loss_layer_coefficients[i] = (
                1.0 / self.vgg.features[num].weight.data.numel()
            )

        print("loss_coefficients:", self.loss_layer_coefficients)
        self.norm = torch.nn.modules.normalization

    def forward(self, inputs: tuple):
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

        batch_size = input.shape[0]
        for b in range(batch_size):

            # TODO Implement correct loss
            for i in range(len(result_gen)):
                res = (
                    result_truth[i][b] - result_gen[i][b]
                ).norm() * self.loss_layer_coefficients[i]
                total_loss += res

            total_loss += (input[b] - truth[b]).norm() * self.loss_direct_coefficient

        del result_gen, result_truth
        # total loss reduction = mean
        return total_loss/batch_size

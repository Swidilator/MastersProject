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


class CircularList:
    def __init__(self, input: int):
        self.len = input
        self.data: list = [1.0 for x in range(input)]
        self.pointer: int = 0

    def update(self, input: float) -> None:
        self.data[self.pointer] = input
        if self.pointer + 1 == self.len:
            self.pointer = 0
        else:
            self.pointer += 1

    def sum(self) -> float:
        return sum(self.data)

    def avg(self) -> float:
        return sum(self.data) / self.len


def get_layer_values(
    self: torch.nn.modules.conv.Conv2d, input: tuple, output: torch.Tensor
) -> None:
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    self.stored_output = output


class PerceptualLossNetwork(modules.Module):
    def __init__(self, input_image_size: tuple, history_len: int):
        super(PerceptualLossNetwork, self).__init__()

        self.vgg: torchvision.models.VGG = torchvision.models.vgg19(
            pretrained=True, progress=True
        )
        self.vgg.eval()
        del self.vgg.classifier, self.vgg.avgpool
        for i in self.vgg.features:
            i.requires_grad = False

        self.norm = torch.nn.modules.normalization

        # self.loss_layer_numel = [0, 0, 0, 0, 0]

        self.loss_layer_numbers: tuple = (2, 7, 12, 21, 30)
        # self.loss_layer_coefficient_bases = [0.0, 0.0, 0.0, 0.0, 0.0]
        # self.loss_direct_input_coefficient_base = [0.0]
        # self.loss_layer_coefficients: list = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.loss_layer_history: list = []
        self.loss_layer_scales: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # Network input element count
        # num_elements_input = 1
        # for val in input_image_size:
        #     num_elements_input *= val
        # self.loss_direct_input_coefficient_base: float = 1.0 / num_elements_input

        # History
        for i in range(len(self.loss_layer_scales)):
            self.loss_layer_history.append(CircularList(history_len))

        # Loss layer coefficient base calculations
        for i, num in enumerate(self.loss_layer_numbers):
            self.vgg.features[num].register_forward_hook(get_layer_values)
            # self.loss_layer_coefficient_bases[i] = (
            #     1.0 / self.vgg.features[num].weight.data.numel()
            # )

        # Set initial loss layer coefficients
        # self.loss_layer_coefficients = self.loss_layer_coefficient_bases
        # self.loss_direct_input_coefficient = self.loss_direct_input_coefficient_base
        # print("loss_coefficients:", self.loss_layer_coefficients)

    def update_lambdas(self):
        avg_list: list = [
            self.loss_layer_history[i].avg()
            for i in range(len(self.loss_layer_history))
        ]
        avg_total: float = sum(avg_list) / len(avg_list)
        # for i, val in enumerate(avg_list):
        #     scale_factor: float = val / avg_total
        #     try:
        #         self.loss_layer_coefficients[i] = self.loss_layer_coefficient_bases[i]/scale_factor
        #     except Exception:
        #         self.loss_direct_input_coefficient = self.loss_direct_input_coefficient_base/scale_factor
        for i, val in enumerate(avg_list):
            scale_factor: float = val / avg_total
            self.loss_layer_scales[i] = 1.0 / scale_factor
        pass

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
                res = (result_truth[i][b] - result_gen[i][b]).norm() * (
                    1.0 / result_truth[i][b].numel()
                )
                self.loss_layer_history[i].update(res.item())
                total_loss += res / self.loss_layer_scales[i]

            input_loss: torch.Tensor = (input[b] - truth[b]).norm() / input[b].numel()
            self.loss_layer_history[-1].update(input_loss.item())
            total_loss += input_loss / self.loss_layer_scales[-1]

        del result_gen, result_truth
        # total loss reduction = mean
        return total_loss / batch_size

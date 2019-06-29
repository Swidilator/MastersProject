import torch
import torchvision
from copy import copy


class perceptual_loss(torch.nn.Module):
    def __init__(self):
        super(perceptual_loss, self).__init__()

        vgg = torchvision.models.vgg19(pretrained=True, progress=True)
        layer_list: list = [
            # 2 7 12 21 30
            copy(vgg.features[2]),
            copy(vgg.features[7]),
            copy(vgg.features[12]),
            copy(vgg.features[21]),
            copy(vgg.features[30])
        ]


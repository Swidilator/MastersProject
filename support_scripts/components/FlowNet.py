import torch
import numpy as np

from dataclasses import dataclass

import os

from support_scripts.flownet2_pytorch.models import FlowNet2


@dataclass
class ArgShim:
    fp16: bool
    rgb_max: float


class FlowNetWrapper(torch.nn.Module):
    def __init__(
            self,
            flownet_checkpoint_folder: str
    ):
        super().__init__()
        # load the state_dict
        dict = torch.load(
            flownet_checkpoint_folder
        )

        args = ArgShim(False, 255.0)
        self.net = FlowNet2(args)
        self.net.load_state_dict(dict["state_dict"])
        for i in self.parameters():
            i.requires_grad = False

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Assumes batch size of 1.

        :param img1: (torch.Tensor): First image in sequence.
        :param img2: (torch.Tensor): Second image in sequence.
        :return: (torch.Tensor): Flow tensor
        """
        with torch.no_grad():
            input_imgs: torch.Tensor = torch.cat((img1, img2), dim=0)
            input_imgs = input_imgs.permute(1, 0, 2, 3).unsqueeze(0)
            out = self.net(input_imgs)
            return out

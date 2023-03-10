import torch
from dataclasses import dataclass


@dataclass
class ArgShim:
    fp16: bool
    rgb_max: float


class FlowNetWrapper(torch.nn.Module):
    def __init__(self, flownet_checkpoint_folder: str):
        """
        Wrapper around flownet 2 to make it easy and convenient to use

        """
        super().__init__()
        from support_scripts.components.flownet2_pytorch.models import (
            FlowNet2 as __FlowNet2__,
        )

        # load the state_dict
        state_dict = torch.load(flownet_checkpoint_folder)

        # Base network takes args, this fills them in
        args = ArgShim(False, 1.0)
        self.net = __FlowNet2__(args)
        self.net.load_state_dict(state_dict["state_dict"])
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

    @staticmethod
    def get_grid(
        batch_size, input_height_width: tuple, device: torch.device
    ) -> torch.Tensor:
        # Taken from vid2vid code, necessary for resample to work
        rows, cols = input_height_width

        hor = torch.linspace(-1.0, 1.0, cols)
        hor.requires_grad = False
        hor = hor.view(1, 1, 1, cols)
        hor = hor.expand(batch_size, 1, rows, cols)
        ver = torch.linspace(-1.0, 1.0, rows)
        ver.requires_grad = False
        ver = ver.view(1, 1, rows, 1)
        ver = ver.expand(batch_size, 1, rows, cols)

        t_grid = torch.cat([hor, ver], 1)
        t_grid.requires_grad = False

        return t_grid.to(device)

    @staticmethod
    def resample(
        image: torch.Tensor, flow: torch.Tensor, grid: torch.Tensor
    ) -> torch.Tensor:
        # Taken from vid2vid, performs correct warping using flow
        b, c, h, w = image.size()
        flow = torch.cat(
            [
                flow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                flow[:, 1:2, :, :] / ((h - 1.0) / 2.0),
            ],
            dim=1,
        )
        final_grid = (grid + flow).permute(0, 2, 3, 1)
        output = torch.nn.functional.grid_sample(image, final_grid, align_corners=True)
        return output

from GAN.Blocks import *


class Discriminator(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(Discriminator, self).__init__()

        self.device = device

        self.input_size: tuple = (70, 70)

        self.final_kernel_size: int = 2

        self.cl1: CCILBlock = CCILBlock(64, 3, first_layer=True)
        self.cl2: CCILBlock = CCILBlock(128, self.cl1.get_output_filter_count, False)
        self.cl3: CCILBlock = CCILBlock(256, self.cl2.get_output_filter_count, False)
        self.cl4: CCILBlock = CCILBlock(512, self.cl3.get_output_filter_count, False)

        self.final_conv: nn.Conv2d = nn.Conv2d(
            self.cl4.get_output_filter_count, 1, self.final_kernel_size
        )
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.cl1(input)
        out: torch.Tensor = self.cl2(out)
        out: torch.Tensor = self.cl3(out)
        out: torch.Tensor = self.cl4(out)
        out = self.final_conv(out)
        out = self.sigmoid(out)
        out = torch.mean(out, axis=(1, 2, 3))

        return out

from GAN.Blocks import *


class FullDiscriminator(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(FullDiscriminator, self).__init__()

        self.device: torch.device = device
        self.input_size: tuple = (70, 70)

        self.discriminator_1: SingleDiscriminator = SingleDiscriminator(self.device)
        self.discriminator_1 = self.discriminator_1.to(self.device)
        self.discriminator_2: SingleDiscriminator = SingleDiscriminator(self.device)
        self.discriminator_2 = self.discriminator_2.to(self.device)
        self.discriminator_3: SingleDiscriminator = SingleDiscriminator(self.device)
        self.discriminator_3 = self.discriminator_3.to(self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        output: torch.Tensor = self.discriminator_1(input).view(-1)

        new_size: tuple = (int(input.shape[2] / 2), int(input.shape[3] / 2))
        input_small = torch.nn.functional.interpolate(
            input=input, size=new_size, mode="bilinear"
        )
        output = torch.cat((output, self.discriminator_2(input_small).view(-1)))

        new_size: tuple = (int(input.shape[2] / 4), int(input.shape[3] / 4))
        input_small = torch.nn.functional.interpolate(
            input=input, size=new_size, mode="bilinear"
        )
        output = torch.cat((output, self.discriminator_3(input_small).view(-1)))

        # output = output / 3

        return output


class SingleDiscriminator(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(SingleDiscriminator, self).__init__()

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
        # out = torch.mean(out, axis=(1, 2, 3))

        return out

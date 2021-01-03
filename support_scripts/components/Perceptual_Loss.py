import torch
import torch.nn.modules as modules
import torchvision
from torchvision import transforms


class CircularList:
    def __init__(self, list_len: int):
        self.len = list_len
        self.data: list = [1.0] * list_len
        self.pointer: int = 0

    def update(self, input_value: float) -> None:
        self.data[self.pointer] = input_value
        if self.pointer + 1 == self.len:
            self.pointer = 0
        else:
            self.pointer += 1

    def sum(self) -> float:
        return sum(self.data)

    def mean(self) -> float:
        return sum(self.data) / self.len


class FeatureNet(modules.Module):
    def __init__(self, model: str):
        """
        Creates and cleans up the base model used in the perceptual loss network.

        """
        super(FeatureNet, self).__init__()
        if model == "VGG":
            feature_network: torch.nn.Sequential = torchvision.models.vgg19(
                pretrained=True, progress=True
            ).features
            loss_layer_numbers: tuple = (4, 9, 14, 23, 32)
        elif model == "MobileNet":
            feature_network: torch.nn.Sequential = torchvision.models.mobilenet_v2(
                pretrained=True, progress=True
            ).features
            loss_layer_numbers: tuple = (4, 6, 8, 13, 17)
        else:
            raise ValueError
        # fmt: off
        self.seq_1 = feature_network[0:loss_layer_numbers[0]].eval()
        self.seq_2 = feature_network[loss_layer_numbers[0]:loss_layer_numbers[1]].eval()
        self.seq_3 = feature_network[loss_layer_numbers[1]:loss_layer_numbers[2]].eval()
        self.seq_4 = feature_network[loss_layer_numbers[2]:loss_layer_numbers[3]].eval()
        self.seq_5 = feature_network[loss_layer_numbers[3]:loss_layer_numbers[4]].eval()
        # fmt: on
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor):
        out1 = self.seq_1(image)
        out2 = self.seq_2(out1)
        out3 = self.seq_3(out2)
        out4 = self.seq_4(out3)
        out5 = self.seq_5(out4)
        return out1, out2, out3, out4, out5


class PerceptualLossNetwork(modules.Module):
    def __init__(
        self,
        base_model: str,
        device: torch.device,
        use_loss_output_image: bool,
        loss_scaling_method: str,
    ):
        super(PerceptualLossNetwork, self).__init__()

        self.device: torch.device = device
        self.base_model: str = base_model
        self.use_loss_output_image: bool = use_loss_output_image
        self.loss_scaling_method: str = loss_scaling_method

        self.feature_network: FeatureNet = FeatureNet(self.base_model)

        self.normalise = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Various scaling methods, official should be used at all times
        if self.loss_scaling_method == "official":
            # Values taken from official source code, no idea how they got them
            self.loss_layer_scales: list = [1.6, 2.3, 1.8, 2.8, 0.08, 1.0]
        elif self.loss_scaling_method == "altered":
            # Similar to VGGLoss values in pix2pixHD
            self.loss_layer_scales: list = [1.28, 0.64, 0.32, 0.16, 0.08, 1.0]
        elif self.loss_scaling_method == "may10":
            # Loss used in the one run that looked like something...
            self.loss_layer_scales: list = [1.0, 1.6, 2.3, 1.8, 2.8, 1.0]
        elif self.loss_scaling_method == "flat":
            self.loss_layer_scales: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            raise ValueError("Invalid value of loss_scaling_method.")

    @staticmethod
    def __calculate_loss(
        gen: torch.Tensor, truth: torch.Tensor, label_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate diversity loss.

        """
        assert len(gen.shape) == 5, "Fake image should be 5D"
        loss = torch.mean(
            label_images * torch.mean(torch.abs(truth - gen), dim=1, keepdim=True),
            dim=(2, 3),
        )
        return loss

    def forward(
        self,
        input_gen_init: torch.Tensor,
        input_truth_init: torch.Tensor,
        input_label: torch.Tensor,
    ):
        # Normalise image data for use in perceptual loss
        input_truth = input_truth_init.clone()
        for b in range(input_truth_init.shape[0]):
            input_truth[b] = self.normalise(input_truth_init[b].clone())

        input_gen = input_gen_init.clone()
        for b in range(input_gen_init.shape[0]):
            for out_img in range(input_gen_init.shape[1]):
                input_gen[b, out_img] = self.normalise(input_gen_init[b, out_img])

        this_batch_size = input_gen.shape[0]

        # Loss function requires multiple images per image, so 5D
        input_truth = input_truth.unsqueeze(1)
        input_label = input_label.unsqueeze(1)

        loss: torch.Tensor = torch.zeros(
            this_batch_size,
            input_gen.shape[1],
            input_label.shape[2],
            device=self.device,
        )

        for b in range(this_batch_size):
            result_truth: tuple = self.feature_network(input_truth[b])
            result_gen: tuple = self.feature_network(input_gen[b])

            # Direct input image comparison
            if self.use_loss_output_image:
                input_loss: torch.Tensor = self.__calculate_loss(
                    input_gen[b], input_truth[b], input_label[b]
                )
                loss[b] += input_loss / self.loss_layer_scales[-1]

            # Feature network layer output comparisons
            for i in range(len(result_gen)):
                label_shape: tuple = tuple(result_truth[i][b].shape[1:])
                label_interpolate = torch.nn.functional.interpolate(
                    input=input_label[b], size=label_shape, mode="nearest"
                )

                layer_loss: torch.Tensor = self.__calculate_loss(
                    result_gen[i],
                    result_truth[i].detach(),
                    label_interpolate,
                )

                loss[b] += layer_loss / self.loss_layer_scales[i]

        # Calculate the minimum loss according to the diversity loss method
        min_loss, _ = torch.min(loss, dim=1)

        min_component = min_loss.sum(dim=1, keepdim=True) * 0.999
        mean_component = loss.mean(dim=1).sum(dim=1, keepdim=True) * 0.001

        total_loss: torch.Tensor = min_component + mean_component

        return torch.mean(total_loss)

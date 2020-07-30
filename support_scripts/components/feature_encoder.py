import torch
from torch import nn
import numpy as np

from support_scripts.components.blocks import EncoderBlock


class FeatureEncoder(nn.Module):
    def __init__(
        self, input_channel_count: int, output_channel_count: int, downsample_count: int
    ):
        super(FeatureEncoder, self).__init__()

        self.input_channel_count: int = input_channel_count
        self.output_channel_count: int = output_channel_count
        self.downsample_count: int = downsample_count

        featuremap_count: int = 16

        encoder: nn.ModuleList = nn.ModuleList()
        decoder: nn.ModuleList = nn.ModuleList()

        # Initial layer
        encoder.append(
            EncoderBlock(
                filter_count=featuremap_count,
                input_channel_count=input_channel_count,
                kernel_size=7,
                stride=1,
                padding_size=3,
                use_reflect_pad=True,
                use_instance_norm=True,
                transpose_conv=False,
                use_relu=True,
            )
        )

        # Downsampling
        for i in range(downsample_count):
            multiplier: int = 2 ** i
            encoder.append(
                EncoderBlock(
                    filter_count=featuremap_count * multiplier * 2,
                    input_channel_count=featuremap_count * multiplier,
                    kernel_size=3,
                    stride=2,
                    padding_size=1,
                    use_reflect_pad=False,
                    use_instance_norm=True,
                    transpose_conv=False,
                    use_relu=True,
                )
            )

        # Upsampling
        for i in range(downsample_count):
            multiplier: int = 2 ** (downsample_count - i)
            decoder.append(
                EncoderBlock(
                    filter_count=(featuremap_count * multiplier) // 2,
                    input_channel_count=featuremap_count * multiplier,
                    kernel_size=3,
                    stride=2,
                    padding_size=1,
                    use_reflect_pad=False,
                    use_instance_norm=True,
                    transpose_conv=True,
                    use_relu=True,
                )
            )

        # Final layer
        decoder.append(
            EncoderBlock(
                filter_count=input_channel_count,
                input_channel_count=featuremap_count,
                kernel_size=7,
                stride=1,
                padding_size=3,
                use_reflect_pad=True,
                use_instance_norm=False,
                transpose_conv=False,
                use_relu=False,
            )
        )

        self.tan_h: nn.Tanh = nn.Tanh()

        self.encoder_model: nn.Sequential = nn.Sequential(*encoder)
        self.decoder_model: nn.Sequential = nn.Sequential(*decoder)

    def forward(
        self, input_tensor: torch.Tensor, instance_map: torch.Tensor
    ) -> torch.Tensor:
        output: torch.Tensor = self.encoder_model(input_tensor)
        output = self.decoder_model(output)
        output = self.tan_h(output).clone()
        # output = FeatureEncoder.average_pool(output, instance_map)
        output = self.average_pool_original(output, instance_map)
        return output

    def encode(self, image_input: torch.Tensor) -> torch.Tensor:
        return self.encoder_model(image_input)

    def decode(self, encoded_input: torch.Tensor) -> torch.Tensor:
        return self.decoder_model(encoded_input)

    def average_pool_original(
        self, decoded_input: torch.Tensor, instance_map: torch.Tensor
    ) -> torch.Tensor:
        # Taken from original source code
        outputs_mean = decoded_input.clone()
        inst_list = np.unique(instance_map.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(outputs_mean.size()[0]):
                indices = torch.nonzero(
                    (instance_map[b : b + 1] == int(i)), as_tuple=False
                )  # n x 4
                for j in range(self.output_channel_count):
                    output_ins = decoded_input[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ] = mean_feat
        return outputs_mean

    @staticmethod
    def average_pool(
        decoded_input: torch.Tensor, instance_map: torch.Tensor
    ) -> torch.Tensor:
        batch_size: int = decoded_input.shape[0]

        # Define tensor to hold output
        output_map: torch.Tensor = torch.zeros_like(decoded_input)

        for bat in range(batch_size):
            # Generate list of unique values per instance map
            instance_unique: torch.Tensor = torch.unique(instance_map[bat])

            for i, val in enumerate(instance_unique):
                # Create filter from instance_stacked for finding individual object
                matching_indices: torch.Tensor = (instance_map[bat][0] == val)

                for channel in range(decoded_input.shape[1]):
                    # Find mean of all values that occur for that individual object
                    matching_vals: torch.Tensor = decoded_input[bat][channel][
                        matching_indices
                    ]
                    mean_val: torch.Tensor = matching_vals.mean()
                    # Set output for individual object to mean value
                    output_map[bat][channel][matching_indices] = mean_val

        # Return the average pooled output
        return output_map

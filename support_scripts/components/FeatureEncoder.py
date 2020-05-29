import torch
from torch import nn

from GAN.Blocks import EncoderBlock


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
                transpose_conv=False,
                use_relu=True,
            )
        )

        self.tan_h: nn.Tanh = nn.Tanh()

        self.encoder_model: nn.Sequential = nn.Sequential(*encoder)
        self.decoder_model: nn.Sequential = nn.Sequential(*decoder)

    def forward(self, input: torch.Tensor, instance_map: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.encoder_model(input)
        output = self.decoder_model(output)
        output = self.tan_h(output).clone()
        output = FeatureEncoder.average_pool(output, instance_map)
        return output

    def encode(self, image_input: torch.Tensor) -> torch.Tensor:
        return self.encoder_model(image_input)

    def decode(self, encoded_input: torch.Tensor) -> torch.Tensor:
        return self.decoder_model(encoded_input)

    @staticmethod
    def average_pool(
        decoded_input: torch.Tensor, instance_map: torch.Tensor
    ) -> torch.Tensor:
        batch_size: int = decoded_input.shape[0]

        # Define tensor to hold output
        output_map: torch.Tensor = torch.zeros_like(decoded_input)

        # Stack instance map into same shape as decoded_input, allowing easy comparison
        # instance_stacked: torch.Tensor = torch.cat(
        #     (instance_map, instance_map, instance_map), dim=1
        # )

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
                # Diagnostic printing
                # print(
                #     "{bat}.{i}. Value: {val}, Mean: {mean}, Count: {count}".format(
                #         bat=bat, i=i, val=val, mean=mean_val, count=output_map[bat][matching_indices[0]].numel()
                #     )
                # )
        # Return the average pooled output
        return output_map

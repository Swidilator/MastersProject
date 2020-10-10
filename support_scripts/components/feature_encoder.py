import torch
from torch import nn
import numpy as np
import pandas as pd
import kmeans_pytorch
from tqdm import tqdm
from os import path, mkdir

from support_scripts.components.blocks import EncoderBlock


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        downsample_count: int,
        device: torch.device,
        model_save_dir: str,
        use_clustered_means: bool,
    ):
        super(FeatureEncoder, self).__init__()

        self.input_channel_count: int = input_channel_count
        self.output_channel_count: int = output_channel_count
        self.downsample_count: int = downsample_count
        self.device: torch.device = device
        self.model_save_dir: str = model_save_dir

        featuremap_count: int = 16

        encoder: nn.ModuleList = nn.ModuleList()
        decoder: nn.ModuleList = nn.ModuleList()

        if use_clustered_means:
            means_file_path = path.join(model_save_dir, "clustered_means.pt")
            self.feature_extractions_sampler: FeatureExtractionsSampler = FeatureExtractionsSampler.from_file(
                means_file_path, self.device
            )

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

    def sample_using_means(self, instance_map: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
        return self.feature_extractions_sampler(instance_map, msk)

    def forward(
        self,
        input_tensor: torch.Tensor,
        instance_map: torch.Tensor,
        saved_features_id: dict = None,
        flip_image: bool = False,
    ) -> torch.Tensor:
        if not saved_features_id:
            output: torch.Tensor = self.encoder_model(input_tensor)
            output = self.decoder_model(output)
            output = self.tan_h(output).clone()
            output = self.average_pool(output, instance_map)
            return output
        else:
            upscale_size: torch.Size = instance_map.shape[2:]
            img_name: str = saved_features_id["name"][0]

            sub_path = path.join(
                self.model_save_dir,
                saved_features_id["split"][0],
                saved_features_id["town"][0],
                f"{img_name}_encoded.pt",
            )

            encoded_img = torch.load(sub_path, map_location=self.device)
            encoded_img = torch.nn.functional.interpolate(
                encoded_img, upscale_size, mode="nearest"
            )
            if flip_image:
                encoded_img = encoded_img.flip(3)
            return encoded_img

    def encode(self, image_input: torch.Tensor) -> torch.Tensor:
        return self.encoder_model(image_input)

    def decode(self, encoded_input: torch.Tensor) -> torch.Tensor:
        return self.decoder_model(encoded_input)

    def average_pool(
        self, decoded_input: torch.Tensor, instance_map: torch.Tensor
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

    def extract_features(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_images: bool = False,
        save_dir: str = None,
    ) -> (torch.Tensor, pd.DataFrame):
        # Extract features

        features: torch.Tensor
        features, _ = self.__extract_raw_feature_values__(
            data_loader, save_images, save_dir
        )

        clustered_means: torch.Tensor = torch.empty(0, 4, device=self.device)

        for i, val in enumerate(
            tqdm(features[:, 0].unique(), desc="Processing raw features")
        ):
            # if val > 1000:
            #     val_class = val // 1000

            subset: torch.Tensor = features[features[:, 0] == val][:, 2:5]

            num_centres: int = (10 if subset.shape[0] >= 10 else subset.shape[0])

            cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
                X=subset,
                num_clusters=num_centres,
                distance="euclidean",
                device=self.device,
            )

            formatted_means: torch.Tensor = torch.empty(
                num_centres, 4, device=self.device
            ).float()
            formatted_means[:, 0] = val
            formatted_means[:, 1:4] = cluster_centers

            clustered_means = torch.cat((clustered_means, formatted_means))

        np_clustered_means: np.ndarray = clustered_means.cpu().numpy()

        output_dataframe = pd.DataFrame(
            data=np.float_(np_clustered_means),
            index=np.arange(0, np_clustered_means.shape[0]),
            columns=["Semantic Class", "Mean_1", "Mean_2", "Mean_3"],
        )
        return clustered_means, output_dataframe

    def __extract_raw_feature_values__(
        self, data_loader: torch.utils.data.DataLoader, save_images: bool, save_dir: str
    ) -> (torch.Tensor, pd.DataFrame):

        self.eval()

        with torch.no_grad():
            output_tensor: torch.Tensor = torch.empty(0, 5, device=self.device)

            for (batch_idx, input_dict,) in enumerate(
                tqdm(data_loader, desc="Extracting raw features")
            ):
                batch_size: int = input_dict["img"].shape[0]
                if batch_size > 1:
                    raise ValueError(
                        "extract_features can/should only be run with batch size 1."
                    )
                real_img = input_dict["img"].to(self.device)
                msk = input_dict["msk"].to(self.device)
                instance_original = input_dict["inst"].to(self.device)
                encoded_img: torch.Tensor = self.__call__(real_img, instance_original)
                if save_images:
                    if save_dir is None:
                        raise ValueError("model_save_dir is None.")
                    # Make dirs
                    if not path.exists(save_dir):
                        mkdir(self.model_save_dir)
                    if not path.exists(
                        sub_path := path.join(
                            self.model_save_dir, input_dict["img_id"]["split"][0]
                        )
                    ):
                        mkdir(sub_path)
                    if not path.exists(
                        sub_path := path.join(sub_path, input_dict["img_id"]["town"][0])
                    ):
                        mkdir(sub_path)

                    img_name: str = input_dict["img_id"]["name"][0]
                    torch.save(
                        encoded_img.to(torch.half),
                        path.join(sub_path, f"{img_name}_encoded.pt"),
                    )

                for bat in range(batch_size):

                    instance_unique: torch.Tensor = torch.unique(instance_original[bat])
                    msk_flat: torch.Tensor = torch.argmax(msk[bat], dim=0)

                    for i, val in enumerate(instance_unique):
                        val_class = val
                        if val > 1000:
                            val_class = val // 1000

                        output_encoding: torch.Tensor = torch.zeros(3)

                        matching_indices_instance: torch.Tensor = (
                            instance_original[bat][0] == val
                        )

                        semantic_class: int = msk_flat[matching_indices_instance].median().int().item()

                        for channel in range(encoded_img.shape[1]):
                            mean_val_instance: torch.Tensor = encoded_img[bat][channel][
                                matching_indices_instance
                            ].mean()

                            output_encoding[channel] = mean_val_instance

                        output_tensor = torch.cat(
                            (
                                output_tensor,
                                torch.tensor(
                                    [
                                        semantic_class,
                                        val_class,
                                        output_encoding[0],
                                        output_encoding[1],
                                        output_encoding[2],
                                    ],
                                    device=self.device,
                                ).unsqueeze(0),
                            )
                        )

        output_table: np.ndarray = output_tensor.cpu().numpy()

        output_dataframe = pd.DataFrame(
            data=np.float_(output_table),
            index=np.arange(0, output_table.shape[0]),
            columns=["Semantic Class", "Class", "Value_1", "Value_2", "Value_3"],
        )

        return (
            output_tensor,
            output_dataframe.sort_values(["Semantic Class"]).round({"Semantic Class": 0}),
        )


class FeatureExtractionsSampler:
    def __init__(self, cluster_means: torch.Tensor, device: torch.device):

        self.clustered_means: torch.Tensor = cluster_means
        self.device: torch.device = device

    @classmethod
    def from_file(cls, feature_extractions_file_path: str, device: torch.device):
        clustered_means: torch.Tensor = torch.load(feature_extractions_file_path)
        return cls(clustered_means, device)

    def __call__(self, instance_map: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:

        # Number of output channels
        num_output_channels: int = 3

        # Batch size handling
        batch_size: int = instance_map.shape[0]

        # Define the output so that it may be filled in gradually
        output_tensor: torch.Tensor = torch.zeros(
            (
                batch_size,
                num_output_channels,
                instance_map.shape[2],
                instance_map.shape[3],
            ),
            device=self.device,
        )

        for batch_no in range(batch_size):
            # Get all unique instances for the given image
            instance_unique: torch.Tensor = torch.unique(instance_map[batch_no])
            msk_flat: torch.Tensor = torch.argmax(msk[batch_no], dim=0)

            # Loop through every unique instance and fill in it's contribution
            for i, val in enumerate(instance_unique):
                val_class = val
                if val > 1000:
                    val_class = val // 1000

                # Generate a boolean tensor matching the location of the unique instance
                matching_indices_instance: torch.Tensor = (
                    instance_map[batch_no][0] == val
                )

                semantic_class: int = msk_flat[matching_indices_instance].median().int().item()

                # Sample a random setting from the clustered means pertaining to the class of the instance
                valid_settings = self.clustered_means[
                    self.clustered_means[:, 0] == semantic_class
                ]
                num_means: int = valid_settings.shape[0]
                index: int = (torch.rand(1) * num_means).int().item()
                random_setting: torch.Tensor = valid_settings[index][1:]

                for j in range(num_output_channels):
                    output_tensor[
                        batch_no, j, matching_indices_instance
                    ] = random_setting[j]

        return output_tensor

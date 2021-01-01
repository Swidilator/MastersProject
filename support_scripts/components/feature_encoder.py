import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import path, mkdir

from typing import List, Optional, Tuple, Union

from support_scripts.components.blocks import EncoderBlock


def get_instance_unique_map_and_flat_mask(
    instance_map: torch.Tensor,
    use_mask_for_instances: bool,
    mask: Optional[torch.Tensor],
):
    assert len(instance_map.shape) == 3, "instance_map.shape != 3"

    if mask is not None:
        assert len(mask.shape) == 3, "mask.shape != 3"

    instance_map = instance_map[0]

    if mask is not None:
        single_mask_flat: Optional[torch.Tensor] = mask.argmax(dim=0).float()
    else:
        single_mask_flat = None

    if not use_mask_for_instances:
        instance_unique: torch.Tensor = torch.unique(instance_map)
        complete_instance_map: torch.Tensor = instance_map
    else:
        assert mask is not None, "instance_from_mask == True, but mask is None"
        complete_instance_map: torch.Tensor = (single_mask_flat.clone() + 1) * -1
        counts = [
            (x.item(), (instance_map == x).sum().item()) for x in instance_map.unique()
        ]
        max_index: int = torch.argmax(torch.tensor(counts)[:, 1]).item()

        for val in instance_map.unique():
            if val != counts[max_index][0]:
                complete_instance_map[(instance_map == val)] = val
        instance_unique: torch.Tensor = torch.unique(complete_instance_map)

    return instance_unique, complete_instance_map, single_mask_flat


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        downsample_count: int,
        device: torch.device,
        model_save_dir: str,
        use_clustered_means: bool,
        use_mask_for_instances: bool,
        num_semantic_classes: int,
    ):
        super(FeatureEncoder, self).__init__()

        self.input_channel_count: int = input_channel_count
        self.output_channel_count: int = output_channel_count
        self.downsample_count: int = downsample_count
        self.device: torch.device = device
        self.model_save_dir: str = model_save_dir
        self.use_mask_for_instances: bool = use_mask_for_instances
        self.num_semantic_classes: int = num_semantic_classes

        featuremap_count: int = 16

        encoder: nn.ModuleList = nn.ModuleList()
        decoder: nn.ModuleList = nn.ModuleList()

        if use_clustered_means:
            means_file_path = path.join(model_save_dir, "clustered_means.pt")
            self.feature_extractions_sampler: FeatureExtractionsSampler = (
                FeatureExtractionsSampler.from_file(
                    means_file_path,
                    self.device,
                    self.use_mask_for_instances,
                    self.num_semantic_classes,
                )
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

    def sample_using_means(
        self,
        instance_map: torch.Tensor,
        msk: torch.Tensor,
        fixed_class_lists: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[list]]]:
        return self.feature_extractions_sampler(instance_map, msk, fixed_class_lists)

    def forward(
        self,
        input_tensor: torch.Tensor,
        instance_map: torch.Tensor,
        saved_features_id: dict = None,
        flip_image: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not saved_features_id:
            output: torch.Tensor = self.encoder_model(input_tensor)
            output = self.decoder_model(output)
            output = self.tan_h(output).clone()
            output = self.average_pool(output, instance_map, mask)
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
        self,
        decoded_input: torch.Tensor,
        instance_map: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size: int = decoded_input.shape[0]

        # Define tensor to hold output
        output_map: torch.Tensor = torch.zeros_like(decoded_input)

        for bat in range(batch_size):
            # Generate list of unique values per instance map

            if self.use_mask_for_instances:
                mask_bat: Optional[torch.Tensor] = mask[bat]
            else:
                mask_bat = None

            (
                instance_unique,
                complete_instance_map,
                _,
            ) = get_instance_unique_map_and_flat_mask(
                instance_map[bat], self.use_mask_for_instances, mask_bat
            )

            for i, val in enumerate(instance_unique):
                # Create filter from instance_stacked for finding individual object
                matching_indices: torch.Tensor = complete_instance_map == val

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

        from sklearn.cluster import KMeans

        features: torch.Tensor
        features, _ = self.__extract_raw_feature_values__(
            data_loader, save_images, save_dir
        )

        clustered_means: torch.Tensor = torch.empty(0, 4, device=self.device)

        for i, semantic_class in enumerate(
            tqdm(features[:, 0].unique(), desc="Processing raw features")
        ):
            subset: torch.Tensor = features[features[:, 0] == semantic_class][:, 1:4]

            num_centres: int = 10 if subset.shape[0] >= 10 else subset.shape[0]

            sk_kmeans = KMeans(10).fit(subset.cpu().numpy())
            sk_centres = torch.tensor(sk_kmeans.cluster_centers_, device=self.device)

            formatted_means: torch.Tensor = torch.empty(
                num_centres, 4, device=self.device
            ).float()
            formatted_means[:, 0] = semantic_class
            formatted_means[:, 1:4] = sk_centres

            clustered_means = torch.cat((clustered_means, formatted_means))

        np_clustered_means: np.ndarray = clustered_means.cpu().numpy()

        output_dataframe = pd.DataFrame(
            data=np.float_(np_clustered_means),
            index=np.arange(0, np_clustered_means.shape[0]),
            columns=["Semantic Class", "Mean_1", "Mean_2", "Mean_3"],
        )
        return clustered_means, output_dataframe

    def __extract_raw_feature_values__(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_images: bool,
        save_dir: str,
    ) -> (torch.Tensor, pd.DataFrame):

        self.eval()

        with torch.no_grad():
            output_tensor: torch.Tensor = torch.empty(0, 4, device=self.device)

            for (
                batch_idx,
                input_dict,
            ) in enumerate(tqdm(data_loader, desc="Extracting raw features")):
                batch_size: int = input_dict["img"].shape[0]
                if batch_size > 1:
                    raise ValueError(
                        "extract_features can/should only be run with batch size 1."
                    )
                real_img = input_dict["img"].to(self.device)
                msk = input_dict["msk"].to(self.device)
                instance_map = input_dict["inst"].to(self.device)
                img_id = input_dict["img_id"]

                if len(real_img.shape) == 5:
                    assert real_img.shape[1] == 1, "num-frames-per-video != 1"

                    real_img = real_img[:, 0]
                    msk = msk[:, 0]
                    instance_map = instance_map[:, 0]
                    img_id = input_dict["img_id"][0]

                encoded_img: torch.Tensor = self.forward(
                    real_img, instance_map, mask=msk
                )

                if save_images:
                    if save_dir is None:
                        raise ValueError("model_save_dir is None.")
                    # Make dirs
                    if not path.exists(save_dir):
                        mkdir(self.model_save_dir)
                    if not path.exists(
                        sub_path := path.join(self.model_save_dir, img_id["split"][0])
                    ):
                        mkdir(sub_path)
                    if not path.exists(
                        sub_path := path.join(sub_path, img_id["town"][0])
                    ):
                        mkdir(sub_path)

                    img_name: str = img_id["name"][0]
                    torch.save(
                        encoded_img.to(torch.half),
                        path.join(sub_path, f"{img_name}_encoded.pt"),
                    )

                for bat in range(batch_size):

                    (
                        instance_unique,
                        complete_instance_map,
                        mask_flat,
                    ) = get_instance_unique_map_and_flat_mask(
                        instance_map[bat], self.use_mask_for_instances, msk[bat]
                    )

                    for i, val in enumerate(instance_unique):

                        output_encoding: torch.Tensor = torch.zeros(3)

                        matching_indices_instance: torch.Tensor = (
                            complete_instance_map == val
                        )

                        semantic_class: int = (
                            mask_flat[matching_indices_instance].median().int().item()
                        )

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
            columns=["Semantic Class", "Value_1", "Value_2", "Value_3"],
        )

        return (
            output_tensor,
            output_dataframe.sort_values(["Semantic Class"]).round(
                {"Semantic Class": 0}
            ),
        )


class FeatureExtractionsSampler:
    def __init__(
        self,
        cluster_means: torch.Tensor,
        device: torch.device,
        use_masks_as_instances: bool,
        num_semantic_classes: int,
    ):

        self.clustered_means: torch.Tensor = cluster_means
        self.device: torch.device = device
        self.use_masks_as_instances: bool = use_masks_as_instances
        self.single_setting_class_list: Optional[list] = None
        self.num_semantic_classes: int = num_semantic_classes
        self.update_single_setting_class_list()

    @classmethod
    def from_file(
        cls,
        feature_extractions_file_path: str,
        device: torch.device,
        use_mask_for_instances: bool,
        num_semantic_classes: int,
    ):
        clustered_means: torch.Tensor = torch.load(feature_extractions_file_path)
        if clustered_means.shape[1] == 5:
            clustered_means = clustered_means[:, 2:5]
        return cls(
            clustered_means, device, use_mask_for_instances, num_semantic_classes
        )

    def update_single_setting_class_list(self):
        single_setting_class_list = []

        num_clustered_classes: int = len(self.clustered_means[:, 0].unique())

        for semantic_class in range(num_clustered_classes):
            valid_settings = self.clustered_means[
                self.clustered_means[:, 0] == semantic_class
            ]
            num_means: int = valid_settings.shape[0]
            # valid_class_counts.append(num_means)
            index: int = (torch.rand(1) * num_means).int().item()
            random_setting: torch.Tensor = valid_settings[index][1:]
            single_setting_class_list.append(random_setting)

        if (classes_diff := self.num_semantic_classes - num_clustered_classes) > 0:
            for i in range(classes_diff):
                single_setting_class_list.append(
                    torch.tensor([0.0, 0.0, 0.0], device=self.device)
                )

        self.single_setting_class_list = single_setting_class_list

    def __call__(
        self,
        instance_map: torch.Tensor,
        msk: torch.Tensor,
        use_single_setting_class_list: bool,
    ) -> torch.Tensor:

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
            (
                instance_unique,
                complete_instance_map,
                mask_flat,
            ) = get_instance_unique_map_and_flat_mask(
                instance_map[batch_no], self.use_masks_as_instances, msk[batch_no]
            )

            # Loop through every unique instance and fill in it's contribution
            for i, val in enumerate(instance_unique):
                # Generate a boolean tensor matching the location of the unique instance
                matching_indices_instance: torch.Tensor = complete_instance_map == val

                semantic_class: int = (
                    mask_flat[matching_indices_instance].median().int().item()
                )

                if use_single_setting_class_list:
                    chosen_setting = self.single_setting_class_list[semantic_class]
                else:
                    valid_settings = self.clustered_means[
                        self.clustered_means[:, 0] == semantic_class
                    ]
                    num_means: int = valid_settings.shape[0]
                    index: int = (torch.rand(1) * num_means).int().item()
                    chosen_setting: torch.Tensor = valid_settings[index]

                for j in range(num_output_channels):
                    output_tensor[
                        batch_no, j, matching_indices_instance
                    ] = chosen_setting[j]

        return output_tensor

    # def sample_using_random_classes(
    #     self, instance_map: torch.Tensor, msk: torch.Tensor
    # ) -> torch.Tensor:
    #
    #     # Number of output channels
    #     num_output_channels: int = 3
    #
    #     # Batch size handling
    #     batch_size: int = instance_map.shape[0]
    #
    #     # Define the output so that it may be filled in gradually
    #     output_tensor: torch.Tensor = torch.zeros(
    #         (
    #             batch_size,
    #             num_output_channels,
    #             instance_map.shape[2],
    #             instance_map.shape[3],
    #         ),
    #         device=self.device,
    #     )
    #
    #     for batch_no in range(batch_size):
    #         # Get all unique instances for the given image
    #         instance_unique: torch.Tensor = torch.unique(instance_map[batch_no])
    #         msk_flat: torch.Tensor = torch.argmax(msk[batch_no], dim=0)
    #
    #         # Loop through every unique instance and fill in it's contribution
    #         for i, val in enumerate(instance_unique):
    #             val_class = val
    #             if val > 1000:
    #                 val_class = val // 1000
    #
    #             # Generate a boolean tensor matching the location of the unique instance
    #             matching_indices_instance: torch.Tensor = (
    #                 instance_map[batch_no][0] == val
    #             )
    #
    #             semantic_class: int = (
    #                 msk_flat[matching_indices_instance].median().int().item()
    #             )
    #
    #             # Sample a random setting from the clustered means pertaining to the class of the instance
    #             valid_settings = self.clustered_means[
    #                 self.clustered_means[:, 0] == semantic_class
    #             ]
    #             num_means: int = valid_settings.shape[0]
    #             index: int = (torch.rand(1) * num_means).int().item()
    #             random_setting: torch.Tensor = valid_settings[index][1:]
    #
    #             for j in range(num_output_channels):
    #                 output_tensor[
    #                     batch_no, j, matching_indices_instance
    #                 ] = random_setting[j]
    #
    #     return output_tensor

from os import path, listdir
from random import random
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip

import pandas as pd

from support_scripts.utils.datasets.dataset_helpers import (
    generate_edge_map,
    TransformManager,
)


class CityScapesVideoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        should_flip: bool,
        subset_size: int,
        output_image_height_width: tuple,
        generated_data: bool,
        num_frames: int,
        frame_offset: Union[int, str],
    ):
        super().__init__()

        assert generated_data, "generated_data is False, but using generated dataset."

        if not path.exists(root):
            raise ValueError("'root' path does not exist.")
        if split not in ["train", "val"]:
            raise ValueError("Invalid 'split' value.")

        self.num_frames: int = num_frames
        self.subset_size: int = subset_size
        self.should_flip: bool = should_flip

        if frame_offset == "random":
            self.frame_offset: int = 0
            self.random_offset: bool = True
        else:
            self.frame_offset: int = int(frame_offset)
            self.random_offset: bool = False

        if self.frame_offset + self.num_frames > 30:
            raise ValueError("num_frames too great for the given frame offset")

        self.num_video_classes: int = 19

        # Can be used to find number of channels segmentation image, includes cruft layer
        self.num_output_segmentation_classes: int = self.num_video_classes + 1

        # Create full image folder paths
        gt_fine_path: str = path.join(root, "gtFine_sequence/", split + "/")
        left_img_8bit_path: str = path.join(root, "leftImg8bit_sequence/", split + "/")

        # Get image directories
        gt_fine_dirs: list = sorted(listdir(gt_fine_path))
        left_img_dirs: list = sorted(listdir(left_img_8bit_path))

        # Helper function for generating paths of each image.
        def make_full_path(img_dir: str, root_dir: str):
            img_list = sorted(listdir(path.join(root_dir, img_dir)))
            split_list = [x[:-4].split("_") for x in img_list]

            if len(split_list[0]) == 4:
                split_list = [[*x, "real"] for x in split_list]

            full_list = [path.join(root_dir, img_dir, x) for x in img_list]
            final_list = [[*split_list[i], y] for i, y in enumerate(full_list)]

            return full_list, final_list

        data_storage: pd.DataFrame = pd.DataFrame(
            columns=["city", "set", "frame", "folder", "type", "full_path"]
        )

        def update_data_storage(data_storage_holder, split_list):
            temp_df: pd.DataFrame = pd.DataFrame(
                columns=["city", "set", "frame", "folder", "type", "full_path"],
                data=split_list,
            )

            data_storage_holder[0] = data_storage_holder[0].append(
                temp_df, ignore_index=True
            )

        data_storage_holder = [data_storage]

        for img_dir in gt_fine_dirs:
            _, out_list = make_full_path(img_dir, gt_fine_path)
            update_data_storage(data_storage_holder, out_list)

        for img_dir in left_img_dirs:
            _, out_list = make_full_path(img_dir, left_img_8bit_path)
            update_data_storage(data_storage_holder, out_list)

        data_storage = data_storage_holder[0]
        data_storage.sort_values(
            ["city", "folder", "set", "frame", "type"], inplace=True
        )

        unique_image_sets: pd.DataFrame = data_storage[
            ["city", "set"]
        ].drop_duplicates()

        if self.subset_size > 0:
            unique_image_sets = unique_image_sets[
                unique_image_sets["set"].astype(np.int) < subset_size
            ]

        def filter_per_image(row, data_storage, output_type: str):
            row_dict = row.to_dict()
            return data_storage[
                (data_storage["city"] == row_dict["city"])
                & (data_storage["set"] == row_dict["set"])
                & (data_storage["type"] == output_type)
            ]["full_path"].values

        print("Splitting data")

        semantic_target_images: list = [
            filter_per_image(row, data_storage, "labelIds")
            for index, row in unique_image_sets.iterrows()
        ]
        color_target_images: list = [
            filter_per_image(row, data_storage, "color")
            for index, row in unique_image_sets.iterrows()
        ]
        instance_target_images: list = [
            filter_per_image(row, data_storage, "instanceIds")
            for index, row in unique_image_sets.iterrows()
        ]
        real_images: list = [
            filter_per_image(row, data_storage, "real")
            for index, row in unique_image_sets.iterrows()
        ]
        print("Done splitting data")

        self.image_list_dict: dict = {
            "semantic": semantic_target_images,
            "color": color_target_images,
            "instance": instance_target_images,
            "real": real_images,
        }

        self.transform_manager = TransformManager(
            output_image_height_width, self.num_video_classes, generated_data=True
        )

        self.targets_mapping: dict = {
            "semantic": "msk",
            "color": "msk_colour",
            "instance": "inst",
            "real": "img",
        }

    @staticmethod
    def get_img_id(img_path):
        # Extract the useful info from the name of the image for use later
        img_id: list = img_path.split("/")[-3:]
        img_id[-1] = "_".join(img_id[-1].split("_")[:3])

        return {"split": img_id[0], "town": img_id[1], "name": img_id[2]}

    @staticmethod
    def unsqueeze_and_concat(input_frames: list) -> torch.Tensor:
        frames: list = [single_frame.unsqueeze(0) for single_frame in input_frames]
        return torch.cat(frames, dim=0)

    def __getitem__(self, item: int) -> dict:

        output_dict: dict = {
            "img": None,
            "img_path": None,
            "img_id": None,
            "img_flipped": None,
            "msk": None,
            "msk_colour": None,
            "inst": None,
            "edge_map": None,
        }

        # Offsets for multiple frames
        if self.random_offset:
            frame_offset: int = int(torch.rand(1).item() * (30 - self.num_frames))
        else:
            assert self.frame_offset + self.num_frames <= 30, "num_frames too great"
            frame_offset = self.frame_offset

        # Flipping frames
        flip_list_sample = self.should_flip and random() > 0.5
        output_dict["img_flipped"] = flip_list_sample

        for target in self.targets_mapping.keys():
            # Get file names
            frame_file_names: list = self.image_list_dict[target][item][
                frame_offset : frame_offset + self.num_frames
            ]

            if target == "real":
                output_dict["img_id"] = [
                    self.get_img_id(single_file_name)
                    for single_file_name in frame_file_names
                ]
                output_dict["img_path"] = list(frame_file_names)

            # Get images
            frames: list = [
                Image.open(single_file_name) for single_file_name in frame_file_names
            ]

            # Flip images if needed
            if flip_list_sample:
                frames = [
                    transforms.functional.hflip(single_frame) for single_frame in frames
                ]

            transformed_frames: list = [
                self.transform_manager.transform_dict[target](single_frame)
                for single_frame in frames
            ]

            if target == "instance":
                edge_map_frames: list = [
                    generate_edge_map(
                        single_frame, output_dict[self.targets_mapping["semantic"]][i]
                    )
                    for i, single_frame in enumerate(transformed_frames)
                ]
                concatenated_edge_map_frames: torch.Tensor = self.unsqueeze_and_concat(
                    edge_map_frames
                )
                output_dict["edge_map"] = concatenated_edge_map_frames

            transformed_frames: list = [
                single_frame for single_frame in transformed_frames
            ]
            concatenated_frames: torch.Tensor = self.unsqueeze_and_concat(
                transformed_frames
            )
            output_dict[self.targets_mapping[target]] = concatenated_frames

        return output_dict

    def __len__(self):
        # Set length of dataset to subset size intelligently
        if (
            self.subset_size == 0
            or len(self.image_list_dict["semantic"]) < self.subset_size
        ):
            return len(self.image_list_dict["semantic"])
        else:
            return self.subset_size


# class CityScapesDatasetVideoWrapper(Dataset):
#     def __init__(
#         self,
#         output_image_height_width: tuple,
#         root: str,
#         split: str,
#         should_flip: bool,
#         subset_size: int,
#         specific_model: str,
#         num_frames: int,
#     ):
#         super(CityScapesDatasetVideoWrapper, self).__init__()
#
#         generated_data: bool = split == "demoVideo"
#
#         self.dataset = CityScapesStandardDataset(
#             output_image_height_width,
#             root,
#             split,
#             should_flip,
#             subset_size,
#             specific_model,
#             generated_data,
#         )
#
#         # Settings
#         self.output_image_height_width = output_image_height_width
#         self.should_flip: bool = should_flip
#         self.subset_size: int = subset_size
#         self.specific_model: str = specific_model
#         self.split = split
#
#         self.num_output_segmentation_classes = (
#             self.dataset.num_output_segmentation_classes
#         )
#
#         self.num_frames: int = num_frames
#
#     def __len__(self):
#         # Set length of dataset to subset size intelligently
#         if self.subset_size == 0 or self.dataset.__len__() < self.subset_size:
#             return self.dataset.__len__() - self.num_frames
#         else:
#             # TODO Small margin for error at the end
#             return self.subset_size
#
#     def __getitem__(self, item: int):
#         # gets item
#         frames = (item + self.num_frames - 1, self.num_frames)
#
#         dicts = self.dataset[frames]
#         if type(dicts) is dict:
#             dicts = [dicts]
#
#         return collate_fn(dicts)

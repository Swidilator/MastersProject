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


class CityScapesVideoDataset2(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        should_flip: bool,
        subset_size: int,
        output_image_height_width: tuple,
        num_frames: int,
        frame_offset: Union[int, str],
    ):
        super().__init__()
        if not path.exists(root):
            raise ValueError("'root' path does not exist.")
        if split not in ["train", "val"]:
            raise ValueError("Invalid 'split' value.")

        self.num_frames: int = num_frames
        self.subset_size: int = subset_size
        self.should_flip: bool = should_flip

        if type(frame_offset) is int:
            self.frame_offset: int = frame_offset
            self.random_offset: bool = False
        elif frame_offset == "random":
            self.frame_offset: int = 0
            self.random_offset: bool = True
        else:
            raise ValueError("Invalid frame_offset value.")

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

        semantic_target_imgs: list = [
            filter_per_image(row, data_storage, "labelIds")
            for index, row in unique_image_sets.iterrows()
        ]
        color_target_imgs: list = [
            filter_per_image(row, data_storage, "color")
            for index, row in unique_image_sets.iterrows()
        ]
        instance_target_imgs: list = [
            filter_per_image(row, data_storage, "instanceIds")
            for index, row in unique_image_sets.iterrows()
        ]
        real_imgs: list = [
            filter_per_image(row, data_storage, "real")
            for index, row in unique_image_sets.iterrows()
        ]
        print("Done splitting data")

        self.image_list_dict: dict = {
            "semantic": semantic_target_imgs,
            "color": color_target_imgs,
            "instance": instance_target_imgs,
            "real": real_imgs,
        }

        self.transform_manager = TransformManager(
            output_image_height_width, self.num_video_classes, True
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
                    generate_edge_map(single_frame)
                    for single_frame in transformed_frames
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


# class CityScapesDataset(Dataset):
#     def __init__(
#         self,
#         output_image_height_width: tuple,
#         root: str,
#         split: str,
#         should_flip: bool,
#         subset_size: int,
#         noise: bool,
#         dataset_features: dict,
#         specific_model: str,
#         use_all_classes: bool = False,
#     ):
#         super(CityScapesDataset, self).__init__()
#
#         # Settings
#         self.output_image_height_width = output_image_height_width
#         self.should_flip: bool = should_flip
#         self.subset_size: int = subset_size
#         self.noise: bool = noise
#         self.specific_model: str = specific_model
#         self.use_all_classes: bool = use_all_classes
#         self.split = split
#
#         # Number of classes in base CityScapes image
#         self.num_cityscape_classes: int = 34
#
#         # Segmentation network only outputs the 19 train classes
#         if self.split == "demoVideo":
#             self.use_all_classes = True
#             self.num_cityscape_classes = 19
#
#         self.used_segmentation_classes = torch.tensor(
#             [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
#             requires_grad=False,
#         )
#         # Can be used to find number of channels segmentation image, includes cruft layer
#         self.num_output_segmentation_classes: int = (
#             len(self.used_segmentation_classes)
#             if not self.use_all_classes
#             else self.num_cityscape_classes
#         ) + 1
#
#         # Set up optional features and defaults
#         self.dataset_features_dict: dict = {
#             "instance_map": False,
#             "instance_map_processed": False,
#         }
#         self.dataset_features_dict.update(dataset_features)
#
#         # Recreation of the normal CityScapes dataset
#         self.dataset: BaseCityScapesDataset = BaseCityScapesDataset(
#             root=root, split=split, target_type=["semantic", "color", "instance"],
#         )
#
#         # Add features based on feature_dict
#         if self.dataset_features_dict["instance_map_processed"]:
#             self.instance_map_processor: InstanceMapProcessor = InstanceMapProcessor()
#
#         if self.specific_model == "pix2pixHD":
#             (
#                 self.mask_resize_transform,
#                 self.image_resize_transform,
#                 self.instance_resize_transform,
#             ) = self.__create_pix2pix_hd_transforms__(output_image_height_width)
#         else:
#             # Image transforms
#             self.image_resize_transform = transforms.Compose(
#                 [
#                     transforms.Lambda(lambda img: img.convert("RGB")),
#                     transforms.Resize(output_image_height_width, Image.BICUBIC),
#                     transforms.Lambda(lambda img: np.array(img)),
#                     transforms.ToTensor(),
#                 ]
#             )
#             self.instance_resize_transform = transforms.Compose(
#                 [
#                     transforms.Resize(output_image_height_width, Image.NEAREST),
#                     transforms.Lambda(
#                         lambda img: torch.tensor(np.array(img)).unsqueeze(0).float()
#                     ),
#                 ]
#             )
#
#             self.mask_resize_transform = transforms.Compose(
#                 [
#                     transforms.Resize(
#                         output_image_height_width,
#                         Image.NEAREST,  # NEAREST as the values are categories and are not continuous
#                     ),
#                     transforms.Lambda(lambda img: np.array(img)),
#                     transforms.ToTensor(),
#                     transforms.Lambda(lambda img: self.__onehot_scatter__(img)),
#                 ]
#             )
#
#     def __getitem__(self, index: Union[int, tuple]):
#
#         if type(index) is tuple:
#             if len(index) == 1:
#                 num_images = 1
#             else:
#                 num_images = index[1]
#             index = index[0]
#         else:
#             num_images = 1
#
#         input_dict_list: list = []
#
#         if (index + 1) - num_images < 0:
#             index = num_images - 1
#
#         for image_no in range((index + 1) - num_images, (index + 1)):
#             # print(image_no)
#             (img, img_path), (msk, msk_colour, instance) = self.dataset.__getitem__(
#                 image_no
#             )
#
#             # Extract the useful info from the name of the image for use later
#             img_id: list = img_path.split("/")[-3:]
#             img_id[-1] = "_".join(img_id[-1].split("_")[:3])
#
#             img_id_dict = {"split": img_id[0], "town": img_id[1], "name": img_id[2]}
#
#             flip_list_sample = self.should_flip and random() > 0.5
#
#             if flip_list_sample:
#                 img = transforms.functional.hflip(img)
#                 msk = transforms.functional.hflip(msk)
#                 msk_colour = transforms.functional.hflip(msk_colour)
#                 instance = transforms.functional.hflip(instance)
#
#             img: torch.Tensor = self.image_resize_transform(img)
#             msk: torch.Tensor = self.mask_resize_transform(msk)
#             msk_colour: torch.Tensor = self.image_resize_transform(msk_colour)
#             instance: Optional[torch.Tensor] = self.instance_resize_transform(instance)
#
#             if self.dataset_features_dict["instance_map_processed"]:
#                 instance_processed: Optional[torch.Tensor]
#                 instance_processed = self.instance_map_processor(instance)
#             else:
#                 instance_processed = torch.empty(0, requires_grad=False)
#
#             if self.noise and torch.rand(1).item() > 0.5:
#                 img = img + torch.normal(0, 0.02, img.size())
#                 img[img > 1] = 1
#                 img[img < -1] = -1
#             if self.noise and torch.rand(1).item() > 0.5:
#                 mean_range: float = (torch.rand(1).item() * 0.2) + 0.7
#                 msk_noise = torch.normal(mean_range, 0.1, msk.size())
#                 msk_noise = msk_noise.int().float()
#                 # print(msk_noise.sum() / self.num_classes)
#                 msk = msk + msk_noise
#
#             if not self.dataset_features_dict["instance_map"]:
#                 instance = torch.empty(0)
#
#             if self.specific_model == "pix2pixHD":
#                 input_dict = {
#                     "label": msk,
#                     "inst": instance,
#                     "image": img,
#                     "feat": torch.empty(0, requires_grad=False),
#                     "path": img_path,
#                 }
#                 # return input_dict
#             else:
#                 input_dict: dict = {
#                     "img": img,
#                     "img_path": img_path,
#                     "img_id": img_id_dict,
#                     "img_flipped": flip_list_sample,
#                     "msk": msk,
#                     "msk_colour": msk_colour,
#                     "inst": instance,
#                     "edge_map": instance_processed,
#                 }
#                 # return input_dict
#
#             input_dict_list.append(input_dict)
#
#         if len(input_dict_list) == 1:
#             return input_dict_list[0]
#         else:
#             return input_dict_list
#
#     def __len__(self):
#         # Set length of dataset to subset size intelligently
#         if self.subset_size == 0 or self.dataset.__len__() < self.subset_size:
#             return self.dataset.__len__()
#         else:
#             return self.subset_size
#
#     def __onehot_scatter__(self, img: torch.Tensor) -> torch.Tensor:
#         input_size: list = list(img.shape)
#         input_size[0] = self.num_cityscape_classes
#         label: torch.Tensor = torch.zeros(input_size)
#
#         # Scale data into integers
#         img = (img * 255).long()
#
#         # Scatter into one-hot format
#         label = label.scatter_(0, img.long(), 1.0)
#
#         # Select layers based on official guidelines if requested
#         if not self.use_all_classes:
#             label = torch.index_select(label, 0, self.used_segmentation_classes)
#
#         # Combine cruft and unlabeled into one layer
#         layer: torch.Tensor = torch.zeros_like(label[0])
#         layer[label.sum(dim=0) == 0] = 1
#
#         label = torch.cat((label, layer.unsqueeze(dim=0)), dim=0)
#         label[-1] = label[-1] + label[0]
#         label[0] = 0
#
#         return label.float()
#
#     def __create_pix2pix_hd_transforms__(self, height_width) -> tuple:
#         # Mask
#         mask_inst_transform_list = [
#             transforms.Resize(height_width, Image.NEAREST),
#             transforms.Lambda(lambda img: np.array(img)),
#             transforms.ToTensor(),
#         ]
#         # Multiply by 255
#         mask_transform = transforms.Compose(
#             [*mask_inst_transform_list, transforms.Lambda(lambda img: img * 255.0)]
#         )
#
#         # Real image
#         real_image_transform_list = [
#             transforms.Lambda(lambda img: img.convert("RGB")),
#             transforms.Resize(height_width, Image.BICUBIC),
#             transforms.Lambda(lambda img: np.array(img)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#         real_image_transform = transforms.Compose(real_image_transform_list)
#
#         # Instance Maps
#         instance_transform = transforms.Compose(mask_inst_transform_list)
#         return mask_transform, real_image_transform, instance_transform
#
#     @staticmethod
#     def __add_remaining_layer__(img: torch.Tensor):
#         layer: torch.Tensor = torch.zeros_like(img[0])
#         layer[img.sum(dim=0) == 0] = 1
#         return torch.cat((img, layer.unsqueeze(dim=0)), dim=0)
#
#
# class CityScapesDemoVideoDataset(Dataset):
#     def __init__(
#         self,
#         output_image_height_width: tuple,
#         root: str,
#         split: str,
#         should_flip: bool,
#         subset_size: int,
#         noise: bool,
#         dataset_features: dict,
#         specific_model: str,
#         num_frames: int,
#         use_all_classes: bool = False,
#     ):
#         super(CityScapesDemoVideoDataset, self).__init__()
#
#         self.dataset = CityScapesDataset(
#             output_image_height_width,
#             root,
#             split,
#             should_flip,
#             subset_size,
#             noise,
#             dataset_features,
#             specific_model,
#             use_all_classes,
#         )
#
#         # Settings
#         self.output_image_height_width = output_image_height_width
#         self.should_flip: bool = should_flip
#         self.subset_size: int = subset_size
#         self.noise: bool = noise
#         self.specific_model: str = specific_model
#         self.use_all_classes: bool = use_all_classes
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
#
#         return self.collate_fn(dicts)
#
#     @staticmethod
#     def collate_fn(data: list, dim=1):
#
#         out_dict: dict = None
#
#         for single_dict in data:
#             if out_dict is None:
#                 out_dict = single_dict
#                 for k_o, v_o in out_dict.items():
#                     if type(v_o) is torch.Tensor:
#                         out_dict.update({k_o: v_o.unsqueeze(0)})
#                     elif type(v_o) in [str, int, dict, bool]:
#                         out_dict.update({k_o: [v_o]})
#                     elif type(v_o) is list:
#                         out_dict.update({k_o: [v_o]})
#             else:
#                 for k_o, v_o in out_dict.items():
#                     v_s = single_dict[k_o]
#                     if type(v_o) is torch.Tensor:
#                         out_dict.update({k_o: torch.cat((v_o, v_s.unsqueeze(0)))})
#                     elif type(v_o) is list:
#                         out_dict.update({k_o: [*v_o, v_s]})
#
#         return out_dict

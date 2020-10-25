import itertools
from os import path, listdir
from random import random
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip

# from support_scripts.components.feature_encoder import FeatureExtractionsSampler


class BaseCityScapesDataset(Dataset):
    def __init__(self, root: str, split: str, target_type: Union[str, tuple, list]):
        super(BaseCityScapesDataset, self).__init__()
        if not path.exists(root):
            raise ValueError("'root' path does not exist.")
        if (
            split != "train"
            and split != "test"
            and split != "val"
            and split != "demoVideo"
        ):
            raise ValueError("Invalid 'split' value.")

        # Target can be a single string, or an iterable, but code requires an iterable
        self.target_type: Union[str, tuple, list] = target_type if type(
            target_type
        ) is list else [target_type]

        # Create full image folder paths
        gt_fine_path: str = path.join(root, "gtFine/", split + "/")
        left_img_8bit_path: str = path.join(root, "leftImg8bit/", split + "/")

        # Get image directories
        gt_fine_dirs: list = sorted(listdir(gt_fine_path))
        left_img_dirs: list = sorted(listdir(left_img_8bit_path))

        # Helper function for generating paths of each image.
        def make_full_path(img_dir: str, root_dir: str):
            img_list = listdir(path.join(root_dir, img_dir))
            img_list = [path.join(root_dir, img_dir, x) for x in img_list]
            return img_list

        # Get images
        gt_fine_imgs: list = sorted(
            list(
                itertools.chain(
                    *[make_full_path(x, gt_fine_path) for x in gt_fine_dirs]
                )
            )
        )
        left_img_imgs: list = sorted(
            list(
                itertools.chain(
                    *[make_full_path(x, left_img_8bit_path) for x in left_img_dirs]
                )
            )
        )

        # Convert image names into absolute paths
        gt_fine_imgs = [path.abspath(path.join(gt_fine_path, x)) for x in gt_fine_imgs]
        self.__left_img_imgs = [
            path.abspath(path.join(left_img_8bit_path, x)) for x in left_img_imgs
        ]

        # Sort gt_fine images into their respective sets
        self.__semantic_target_imgs: list = [x for x in gt_fine_imgs if "labelIds" in x]
        self.__color_target_imgs: list = [x for x in gt_fine_imgs if "color" in x]
        self.__instance_target_imgs: list = [
            x for x in gt_fine_imgs if "instanceIds" in x
        ]

        # Make sure that the dataset is complete
        assert (
            len(self.__left_img_imgs) == len(self.__semantic_target_imgs)
            and len(self.__left_img_imgs) == len(self.__color_target_imgs)
            and len(self.__left_img_imgs) == len(self.__instance_target_imgs)
        )

    def __getitem__(self, item) -> tuple:
        # targets = ["semantic", "color", "instance"]
        output: list = []
        for target_type in self.target_type:
            if target_type == "semantic":
                output.append(Image.open(self.__semantic_target_imgs[item]))
            elif target_type == "color":
                output.append(Image.open(self.__color_target_imgs[item]))
            elif target_type == "instance":
                output.append(Image.open(self.__instance_target_imgs[item]))

        # Return single image if only one target type, else tuple
        if len(output) == 1:
            return (
                (Image.open(self.__left_img_imgs[item]), self.__left_img_imgs[item]),
                output[0],
            )
        else:
            return (
                (Image.open(self.__left_img_imgs[item]), self.__left_img_imgs[item]),
                tuple(output),
            )

    def __len__(self):
        return len(self.__left_img_imgs)


class CityScapesDataset(Dataset):
    def __init__(
        self,
        output_image_height_width: tuple,
        root: str,
        split: str,
        should_flip: bool,
        subset_size: int,
        noise: bool,
        dataset_features: dict,
        specific_model: str,
        use_all_classes: bool = False,
    ):
        super(CityScapesDataset, self).__init__()

        # Settings
        self.output_image_height_width = output_image_height_width
        self.should_flip: bool = should_flip
        self.subset_size: int = subset_size
        self.noise: bool = noise
        self.specific_model: str = specific_model
        self.use_all_classes: bool = use_all_classes
        self.split = split

        # Number of classes in base CityScapes image
        self.num_cityscape_classes: int = 34

        # Segmentation network only outputs the 19 train classes
        if self.split == "demoVideo":
            self.use_all_classes = True
            self.num_cityscape_classes = 20

        self.used_segmentation_classes = torch.tensor(
            [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
            requires_grad=False,
        )
        # Can be used to find number of channels segmentation image, includes cruft layer
        self.num_segmentation_output_channels: int = len(
            self.used_segmentation_classes
        ) + 1

        # Set up optional features and defaults
        self.dataset_features_dict: dict = {
            "instance_map": False,
            "instance_map_processed": False,
        }
        self.dataset_features_dict.update(dataset_features)

        self.num_output_classes: int = (
            self.num_segmentation_output_channels
            if not self.use_all_classes
            else self.num_cityscape_classes + 1
        )

        # Recreation of the normal CityScapes dataset
        self.dataset: BaseCityScapesDataset = BaseCityScapesDataset(
            root=root, split=split, target_type=["semantic", "color", "instance"],
        )

        # Add features based on feature_dict
        if self.dataset_features_dict["instance_map_processed"]:
            self.instance_map_processor: InstanceMapProcessor = InstanceMapProcessor()

        if self.specific_model == "pix2pixHD":
            (
                self.mask_resize_transform,
                self.image_resize_transform,
                self.instance_resize_transform,
            ) = self.__create_pix2pix_hd_transforms__(output_image_height_width)
        else:
            # Image transforms
            self.image_resize_transform = transforms.Compose(
                [
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    transforms.Resize(output_image_height_width, Image.BICUBIC),
                    transforms.Lambda(lambda img: np.array(img)),
                    transforms.ToTensor(),
                ]
            )
            self.instance_resize_transform = transforms.Compose(
                [
                    transforms.Resize(output_image_height_width, Image.NEAREST),
                    transforms.Lambda(lambda img: np.array(img)),
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda img: img.float()
                        * (255 if self.split == "demoVideo" else 1)
                    ),
                ]
            )

            self.mask_resize_transform = transforms.Compose(
                [
                    transforms.Resize(
                        output_image_height_width,
                        Image.NEAREST,  # NEAREST as the values are categories and are not continuous
                    ),
                    transforms.Lambda(lambda img: np.array(img)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda img: self.__onehot_scatter__(img)),
                ]
            )

    def __getitem__(self, index: Union[int, tuple]):

        if type(index) is tuple:
            if len(index) == 1:
                num_images = 1
            else:
                num_images = index[1]
            index = index[0]
        else:
            num_images = 1

        input_dict_list: list = []

        if (index + 1) - num_images < 0:
            index = num_images - 1

        for image_no in range((index + 1) - num_images, (index + 1)):
            # print(image_no)
            (img, img_path), (msk, msk_colour, instance) = self.dataset.__getitem__(
                image_no
            )

            # Extract the useful info from the name of the image for use later
            img_id: list = img_path.split("/")[-3:]
            img_id[-1] = "_".join(img_id[-1].split("_")[:3])

            img_id_dict = {"split": img_id[0], "town": img_id[1], "name": img_id[2]}

            should_flip = random() > 0.5

            if self.should_flip and should_flip:
                img = transforms.functional.hflip(img)
                msk = transforms.functional.hflip(msk)
                msk_colour = transforms.functional.hflip(msk_colour)
                instance = transforms.functional.hflip(instance)

            img: torch.Tensor = self.image_resize_transform(img)
            msk: torch.Tensor = self.mask_resize_transform(msk)
            msk_colour: torch.Tensor = self.image_resize_transform(msk_colour)
            instance: Optional[torch.Tensor] = self.instance_resize_transform(instance)

            if self.dataset_features_dict["instance_map_processed"]:
                instance_processed: Optional[torch.Tensor]
                instance_processed = self.instance_map_processor(instance)
            else:
                instance_processed = torch.empty(0, requires_grad=False)

            if self.noise and torch.rand(1).item() > 0.5:
                img = img + torch.normal(0, 0.02, img.size())
                img[img > 1] = 1
                img[img < -1] = -1
            if self.noise and torch.rand(1).item() > 0.5:
                mean_range: float = (torch.rand(1).item() * 0.2) + 0.7
                msk_noise = torch.normal(mean_range, 0.1, msk.size())
                msk_noise = msk_noise.int().float()
                # print(msk_noise.sum() / self.num_classes)
                msk = msk + msk_noise

            if not self.dataset_features_dict["instance_map"]:
                instance = torch.empty(0)

            if self.specific_model == "pix2pixHD":
                input_dict = {
                    "label": msk,
                    "inst": instance,
                    "image": img,
                    "feat": torch.empty(0, requires_grad=False),
                    "path": img_path,
                }
                # return input_dict
            else:
                input_dict: dict = {
                    "img": img,
                    "img_path": img_path,
                    "img_id": img_id_dict,
                    "img_flipped": should_flip,
                    "msk": msk,
                    "msk_colour": msk_colour,
                    "inst": instance,
                    "edge_map": instance_processed,
                }
                # return input_dict

            input_dict_list.append(input_dict)

        if len(input_dict_list) == 1:
            return input_dict_list[0]
        else:
            return input_dict_list

    def __len__(self):
        # Set length of dataset to subset size intelligently
        if self.subset_size == 0 or self.dataset.__len__() < self.subset_size:
            return self.dataset.__len__()
        else:
            return self.subset_size

    def __onehot_scatter__(self, img: torch.Tensor) -> torch.Tensor:
        input_size: list = list(img.shape)
        input_size[0] = self.num_cityscape_classes
        label: torch.Tensor = torch.zeros(input_size)

        # Scale data into integers
        img = (img * 255).long()

        # Scatter into one-hot format
        label = label.scatter_(0, img.long(), 1.0)

        # Select layers based on official guidelines if requested
        label = (
            torch.index_select(label, 0, self.used_segmentation_classes)
            if not self.use_all_classes
            else label
        )
        if self.split != "demoVideo":
            layer: torch.Tensor = torch.zeros_like(label[0])
            layer[label.sum(dim=0) == 0] = 1

            label = torch.cat((label, layer.unsqueeze(dim=0)), dim=0)

        return label.float()

    def __create_pix2pix_hd_transforms__(self, height_width) -> tuple:
        # Mask
        mask_inst_transform_list = [
            transforms.Resize(height_width, Image.NEAREST),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
        ]
        # Multiply by 255
        mask_transform = transforms.Compose(
            [*mask_inst_transform_list, transforms.Lambda(lambda img: img * 255.0)]
        )

        # Real image
        real_image_transform_list = [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(height_width, Image.BICUBIC),
            transforms.Lambda(lambda img: np.array(img)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        real_image_transform = transforms.Compose(real_image_transform_list)

        # Instance Maps
        instance_transform = transforms.Compose(mask_inst_transform_list)
        return mask_transform, real_image_transform, instance_transform

    @staticmethod
    def __add_remaining_layer__(img: torch.Tensor):
        layer: torch.Tensor = torch.zeros_like(img[0])
        layer[img.sum(dim=0) == 0] = 1
        return torch.cat((img, layer.unsqueeze(dim=0)), dim=0)


class InstanceMapProcessor:
    def __init__(self):
        cross_element: list = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        cross_element_tensor: torch.Tensor = torch.tensor(
            cross_element, requires_grad=False, dtype=torch.float32
        )
        self.object_separator: torch.nn.Conv2d = torch.nn.Conv2d(1, 1, 3, 1, bias=False)
        self.object_separator.weight.data = cross_element_tensor[(None,) * 2]
        self.object_separator.weight.requires_grad = False

    def __call__(self, instance_map: torch.Tensor):
        assert len(instance_map.shape) == 3, "Invalid tensor shape"
        assert instance_map.shape[0] == 1, "Too many image channels"

        with torch.no_grad():
            edge: torch.Tensor = self.object_separator(instance_map[(None,)])
            edge_border: torch.Tensor = (edge != 0).float().squeeze(0)
            edge_shape: torch.Size = edge_border.shape
            edge_shape = torch.Size((1, edge_shape[1] + 2, edge_shape[2] + 2))
            edge_border_pad: torch.Tensor = torch.zeros(edge_shape)
            edge_border_pad[0, 1:-1, 1:-1] = edge_border
            return edge_border_pad

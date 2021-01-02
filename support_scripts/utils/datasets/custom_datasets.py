import itertools
from os import path, listdir
from random import random
from typing import Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip

from support_scripts.utils.datasets.dataset_helpers import (
    TransformManager,
    generate_edge_map,
    collate_fn,
)


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
        self.target_type: Union[str, tuple, list] = (
            target_type if type(target_type) is list else [target_type]
        )

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


class CityScapesStandardDataset(Dataset):
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
        super(CityScapesStandardDataset, self).__init__()

        assert num_frames == 1, "num_frames > 1, but using standard dataset."
        assert (
            frame_offset == "random"
        ), "frame_offset != random, but using standard dataset."
        if split == "demoVideo":
            assert (
                generated_data
            ), "generated_data is False, but using demoVideo dataset."
        else:
            assert (
                not generated_data
            ), "generated_data is True, but using standard dataset."

        # Settings
        self.output_image_height_width = output_image_height_width
        self.should_flip: bool = should_flip
        self.subset_size: int = subset_size
        self.generated_data: bool = generated_data
        self.split = split

        # Number of classes in base CityScapes image
        num_cityscape_classes: int = 34
        if self.generated_data:
            num_cityscape_classes = 19

        self.transform_manager = TransformManager(
            output_image_height_width,
            num_cityscape_classes,
            generated_data=self.generated_data,
        )

        # Can be used to find number of channels segmentation image, includes cruft layer
        self.num_output_segmentation_classes: int = (
            self.transform_manager.num_output_segmentation_classes
        )

        # Recreation of the normal CityScapes dataset
        self.dataset: BaseCityScapesDataset = BaseCityScapesDataset(
            root=root,
            split=split,
            target_type=["semantic", "color", "instance"],
        )

    def __getitem__(self, index: int) -> dict:
        # print(image_no)
        (img, img_path), (msk, msk_colour, instance) = self.dataset.__getitem__(index)
        # Extract the useful info from the name of the image for use later
        img_id: list = img_path.split("/")[-3:]
        img_id[-1] = "_".join(img_id[-1].split("_")[:3])

        img_id_dict = {"split": img_id[0], "town": img_id[1], "name": img_id[2]}

        flip_list_sample = self.should_flip and random() > 0.5

        if flip_list_sample:
            img = transforms.functional.hflip(img)
            msk = transforms.functional.hflip(msk)
            msk_colour = transforms.functional.hflip(msk_colour)
            instance = transforms.functional.hflip(instance)

        img: torch.Tensor = self.transform_manager.transform_dict["real"](img)
        msk: torch.Tensor = self.transform_manager.transform_dict["semantic"](msk)
        msk_colour: torch.Tensor = self.transform_manager.transform_dict["color"](
            msk_colour
        )
        instance: Optional[torch.Tensor] = self.transform_manager.transform_dict[
            "instance"
        ](instance)
        edge_map: Optional[torch.Tensor] = generate_edge_map(
            instance, msk if self.generated_data else None
        )

        input_dict: dict = {
            "img": img,
            "img_path": img_path,
            "img_id": img_id_dict,
            "img_flipped": flip_list_sample,
            "msk": msk,
            "msk_colour": msk_colour,
            "inst": instance,
            "edge_map": edge_map,
        }

        return collate_fn([input_dict])

    def __len__(self):
        # Set length of dataset to subset size intelligently
        if self.subset_size == 0 or self.dataset.__len__() < self.subset_size:
            return self.dataset.__len__()
        else:
            return self.subset_size

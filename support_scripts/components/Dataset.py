import torch
from os import path, listdir
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision.datasets import Cityscapes
from torchvision import transforms
from torchvision.transforms.functional import hflip
from PIL import Image
from random import random
from typing import Optional, Union
import itertools

from support_scripts import utils


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

        self.target_type: Union[str, tuple, list] = target_type if type(
            target_type
        ) is list else [target_type]

        # Create full image folder paths
        gt_fine_path: str = path.join(root, "gtFine/", split + "/")
        left_img_8bit_path: str = path.join(root, "leftImg8bit/", split + "/")

        # Get image directories
        gt_fine_dirs: list = sorted(listdir(gt_fine_path))
        left_img_dirs: list = sorted(listdir(left_img_8bit_path))

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

        self.__semantic_target_imgs: list = [x for x in gt_fine_imgs if "labelIds" in x]
        self.__color_target_imgs: list = [x for x in gt_fine_imgs if "color" in x]
        self.__instance_target_imgs: list = [
            x for x in gt_fine_imgs if "instanceIds" in x
        ]

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
        if len(output) == 1:
            return Image.open(self.__left_img_imgs[item]), output[0]
        else:
            return Image.open(self.__left_img_imgs[item]), tuple(output)

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
    ):
        super(CityScapesDataset, self).__init__()

        # Settings
        self.output_image_height_width = output_image_height_width
        self.should_flip: bool = should_flip
        self.subset_size: int = subset_size
        self.noise: bool = noise

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
            "feature_extractions": {"use": False, "file_path": None},
        }
        self.dataset_features_dict.update(dataset_features)

        # Number of classes in base CityScapes image
        num_cityscape_classes: int = 34

        # Base CityScapes dataset
        # self.dataset: Cityscapes = Cityscapes(
        #     root=root,
        #     split=split,
        #     mode="fine",
        #     target_type=["semantic", "color", "instance"],
        # )

        self.dataset: BaseCityScapesDataset = BaseCityScapesDataset(
            root=root, split=split, target_type=["semantic", "color", "instance"],
        )

        # test1, (test2, test3, test4) = self.dataset[0]

        # Add features based on feature_dict
        if self.dataset_features_dict["feature_extractions"]["use"]:
            if self.dataset_features_dict["feature_extractions"]["file_path"]:
                self.feature_extractions_sampler = FeatureExtractionsSampler.from_file(
                    self.dataset_features_dict["feature_extractions"]["file_path"]
                )
            else:
                raise ValueError(
                    'dataset_features["feature_extractions"]["file_path"] cannot be None.'
                )
        if self.dataset_features_dict["instance_map_processed"]:
            self.instance_map_processor: InstanceMapProcessor = InstanceMapProcessor()

        # Image transforms
        self.image_resize_transform = transforms.Compose(
            [
                transforms.Resize(output_image_height_width, Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.instance_resize_transform = transforms.Compose(
            [
                transforms.Resize(output_image_height_width, Image.NEAREST),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float()),
            ]
        )
        self.mask_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    output_image_height_width,
                    Image.NEAREST,  # NEAREST as the values are categories and are not continuous
                ),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).long()[0]),
                transforms.Lambda(lambda x: one_hot(x, num_cityscape_classes)),
                transforms.Lambda(
                    lambda x: torch.index_select(x, 2, self.used_segmentation_classes)
                ),
                transforms.Lambda(lambda x: x.transpose(0, 2).transpose(1, 2)),
                transforms.Lambda(
                    lambda x: CityScapesDataset.__add_remaining_layer__(x)
                ),
                transforms.Lambda(lambda x: x.float()),
            ]
        )

    def __getitem__(self, index: int, output_feature_extractions: bool = False):
        img, (msk, msk_colour, instance) = self.dataset.__getitem__(index)
        img = img
        msk = msk
        msk_colour = msk_colour
        instance = instance

        flip: bool = random()

        if self.should_flip and flip > 0.5:
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
            instance_pil: Optional[Image.Image]
            instance_processed, instance_pil = self.instance_map_processor(instance)
        else:
            instance_processed, instance_pil = torch.empty(0), None

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

        # If the feature is not being used, the output will be None
        if self.dataset_features_dict["feature_extractions"]["use"]:
            feature_selection: Optional[torch.Tensor] = (
                self.feature_extractions_sampler(msk, instance)
            )
        else:
            feature_selection: Optional[torch.Tensor] = torch.empty(
                0, requires_grad=False
            )

        if not self.dataset_features_dict["instance_map"]:
            instance = torch.empty(0)

        return img, msk, msk_colour, instance, instance_processed, feature_selection

    def __len__(self):
        # Set length of dataset to subset size intelligently
        if self.subset_size == 0 or self.dataset.__len__() < self.subset_size:
            return self.dataset.__len__()
        else:
            return self.subset_size

    @staticmethod
    def __add_remaining_layer__(x: torch.Tensor):
        layer: torch.Tensor = torch.zeros_like(x[0])
        layer[x.sum(dim=0) == 0] = 1
        return torch.cat((x, layer.unsqueeze(dim=0)), dim=0)

    def set_clustered_means(self, clustered_means: torch.Tensor):
        if self.feature_extractions_sampler:
            self.feature_extractions_sampler.clustered_means = clustered_means
        else:
            self.feature_extractions_sampler = FeatureExtractionsSampler(
                clustered_means
            )


class InstanceMapProcessor:
    def __init__(self):

        cross_element: list = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        cross_element_tensor: torch.Tensor = torch.tensor(
            cross_element, requires_grad=False, dtype=torch.float32
        )
        self.object_separator: torch.nn.Conv2d = torch.nn.Conv2d(1, 1, 3, 1, bias=False)
        self.object_separator.weight.data = cross_element_tensor[(None,) * 2]
        self.object_separator.weight.requires_grad = False

        self.pil_transform = transforms.Compose([transforms.ToPILImage()])

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
            return edge_border_pad, self.pil_transform(edge_border_pad)


class FeatureExtractionsSampler:
    def __init__(self, cluster_means: torch.Tensor):

        self.clustered_means = cluster_means

    @classmethod
    def from_file(cls, feature_extractions_file_path: str):
        import pickle

        with open(feature_extractions_file_path, "rb") as f:
            clustered_means: torch.Tensor = pickle.load(f)
        return cls(clustered_means)

    def __call__(self, msk: torch.Tensor, instance_map: torch.Tensor) -> torch.Tensor:
        # Number of output channels
        num_output_channels: int = 3

        # Get all unique instances for the given image
        instance_unique: torch.Tensor = torch.unique(instance_map)

        # Flatten the one-hot encoded mask into a single channel image
        msk_argmax: torch.Tensor = torch.argmax(msk, dim=0).float()

        # Define the output so that it may be filled in gradually
        output_tensor: torch.Tensor = torch.zeros(
            (num_output_channels, instance_map.shape[1], instance_map.shape[2])
        )
        # Loop through every unique instance and fill in it's contribution
        for i, val in enumerate(instance_unique):

            # Generate a boolean tensor matching the location of the unique instance
            matching_indices_instance: torch.Tensor = (instance_map[0] == val)

            # Extract the class that the unique instance belongs to
            matching_class: torch.tensor = msk_argmax[
                matching_indices_instance
            ].mean().round()

            # Sample a random setting from the clustered means pertaining to the class of the instance
            valid_settings = self.clustered_means[
                self.clustered_means[:, 0] == matching_class
            ]
            num_means: int = valid_settings.shape[0]
            index: int = ((torch.rand(1) * num_means).int()).item()
            random_setting: torch.Tensor = valid_settings[index][1:]

            for j in range(num_output_channels):
                output_tensor[j, matching_indices_instance] = random_setting[j].item()

        return output_tensor

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Optional


class TransformManager:
    def __init__(
        self, output_image_height_width, num_cityscape_classes, generated_data: bool
    ):
        self.one_hot_scatter: OneHotScatter = OneHotScatter(
            num_cityscape_classes, generated_data
        )

        self.num_output_segmentation_classes: int = (
            self.one_hot_scatter.num_output_segmentation_classes
        )

        image_resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(output_image_height_width, Image.BICUBIC),
                transforms.Lambda(lambda img: np.array(img)),
                transforms.ToTensor(),
            ]
        )

        mask_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    output_image_height_width,
                    Image.NEAREST,  # NEAREST as the values are categories and are not continuous
                ),
                transforms.Lambda(lambda img: np.array(img)),
                transforms.ToTensor(),
                transforms.Lambda(lambda img: self.one_hot_scatter(img=img)),
            ]
        )

        instance_resize_transform = transforms.Compose(
            [
                transforms.Resize(output_image_height_width, Image.NEAREST),
                transforms.Lambda(
                    lambda img: torch.tensor(np.array(img)).unsqueeze(0).float()
                ),
            ]
        )

        self.transform_dict: dict = {
            "semantic": mask_resize_transform,
            "color": image_resize_transform,
            "instance": instance_resize_transform,
            "real": image_resize_transform,
        }


class OneHotScatter:
    def __init__(self, num_cityscape_classes, generated_data: bool):
        self.num_cityscape_classes: int = num_cityscape_classes
        self.generated_data: bool = generated_data

        self.used_segmentation_classes = torch.tensor(
            [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
            requires_grad=False,
        ).long()

        self.num_output_segmentation_classes: int = (
            len(self.used_segmentation_classes) + 1
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        input_size: list = list(img.shape)
        input_size[0] = self.num_cityscape_classes
        label: torch.Tensor = torch.zeros(input_size)

        # Scale data into integers
        img = (img * 255).long()

        # Scatter into one-hot format
        label = label.scatter_(0, img.long(), 1.0)

        # Select layers based on official guidelines if requested
        if not self.generated_data:
            label = torch.index_select(label, 0, self.used_segmentation_classes)

        # Combine cruft and unlabeled into one layer
        layer: torch.Tensor = torch.zeros_like(label[0])
        layer[label.sum(dim=0) == 0] = 1

        label = torch.cat((label, layer.unsqueeze(dim=0)), dim=0)
        # label[-1] = label[-1] + label[0]
        # label[0] = 0

        return label.float()


def generate_edge_map(instance_map: torch.Tensor, mask: torch.Tensor = None):
    assert len(instance_map.shape) == 3, "Invalid tensor shape"
    assert instance_map.shape[0] == 1, "Too many image channels"

    cross_element_tensor: torch.Tensor = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], requires_grad=False, dtype=torch.float32
    )
    with torch.no_grad():
        edge = torch.nn.functional.conv2d(
            instance_map.unsqueeze(0), cross_element_tensor[(None,) * 2]
        )
        # edge: torch.Tensor = self.object_separator(instance_map[(None,)])
        edge_border: torch.Tensor = (edge != 0).float().squeeze(0)
        edge_shape: torch.Size = edge_border.shape
        edge_shape = torch.Size((1, edge_shape[1] + 2, edge_shape[2] + 2))
        edge_border_pad: torch.Tensor = torch.zeros(edge_shape)
        edge_border_pad[0, 1:-1, 1:-1] = edge_border

        if mask is not None:
            mask_flat = torch.argmax(mask, dim=0, keepdim=True).float()
            edge_mask = torch.nn.functional.conv2d(
                mask_flat.unsqueeze(0), cross_element_tensor[(None,) * 2]
            )
            # edge: torch.Tensor = self.object_separator(instance_map[(None,)])
            edge_mask_border: torch.Tensor = (edge_mask != 0).float().squeeze(0)
            edge_mask_shape: torch.Size = edge_mask_border.shape
            edge_mask_shape = torch.Size(
                (1, edge_mask_shape[1] + 2, edge_mask_shape[2] + 2)
            )
            edge_mask_border_pad: torch.Tensor = torch.zeros(edge_mask_shape)
            edge_mask_border_pad[0, 1:-1, 1:-1] = edge_mask_border

            edge_border_pad = (edge_border_pad + edge_mask_border_pad).clamp(0, 1)

        return edge_border_pad


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

        with torch.no_grad():
            edge: torch.Tensor = self.object_separator(instance_map[(None,)])
            edge_border: torch.Tensor = (edge != 0).float().squeeze(0)
            edge_shape: torch.Size = edge_border.shape
            edge_shape = torch.Size((1, edge_shape[1] + 2, edge_shape[2] + 2))
            edge_border_pad: torch.Tensor = torch.zeros(edge_shape)
            edge_border_pad[0, 1:-1, 1:-1] = edge_border
            return edge_border_pad


def collate_fn(data: list, dim=1):

    out_dict: Optional[dict] = None

    for single_dict in data:
        if out_dict is None:
            out_dict = single_dict
            for k_o, v_o in out_dict.items():
                if type(v_o) is torch.Tensor:
                    out_dict.update({k_o: v_o.unsqueeze(0)})
                elif type(v_o) in [str, int, dict, bool]:
                    out_dict.update({k_o: [v_o]})
                elif type(v_o) is list:
                    out_dict.update({k_o: [v_o]})
        else:
            for k_o, v_o in out_dict.items():
                v_s = single_dict[k_o]
                if type(v_o) is torch.Tensor:
                    out_dict.update({k_o: torch.cat((v_o, v_s.unsqueeze(0)))})
                elif type(v_o) is list:
                    out_dict.update({k_o: [*v_o, v_s]})

    return out_dict

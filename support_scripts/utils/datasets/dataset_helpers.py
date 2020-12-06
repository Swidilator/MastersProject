import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class TransformManager:
    def __init__(
        self, output_image_height_width, num_cityscape_classes, use_all_classes
    ):
        one_hot_scatter: OneHotScatter = OneHotScatter(
            num_cityscape_classes, use_all_classes
        )

        # used_segmentation_classes = torch.tensor(
        #     [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        #     requires_grad=False,
        # )
        # Image transforms
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
                transforms.Lambda(lambda img: one_hot_scatter(img=img)),
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
    def __init__(self, num_cityscape_classes, use_all_classes):
        self.num_cityscape_classes: int = num_cityscape_classes
        self.use_all_classes: bool = use_all_classes
        self.used_segmentation_classes = torch.tensor(
            [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
            requires_grad=False,
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
        if not self.use_all_classes:
            label = torch.index_select(label, 0, self.used_segmentation_classes)

        # Combine cruft and unlabeled into one layer
        layer: torch.Tensor = torch.zeros_like(label[0])
        layer[label.sum(dim=0) == 0] = 1

        label = torch.cat((label, layer.unsqueeze(dim=0)), dim=0)
        label[-1] = label[-1] + label[0]
        label[0] = 0

        return label.float()


def generate_edge_map(instance_map: torch.Tensor):
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
        return edge_border_pad


def create_transforms(
    output_image_height_width, num_cityscape_classes, use_all_classes,
):
    used_segmentation_classes = torch.tensor(
        [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        requires_grad=False,
    )
    # Image transforms
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
            transforms.Lambda(
                lambda img: onehot_scatter(
                    img=img,
                    num_cityscape_classes=num_cityscape_classes,
                    use_all_classes=use_all_classes,
                    used_segmentation_classes=used_segmentation_classes,
                )
            ),
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

    transform_dict: dict = {
        "semantic": mask_resize_transform,
        "color": image_resize_transform,
        "instance": instance_resize_transform,
        "real": image_resize_transform,
    }

    return transform_dict


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

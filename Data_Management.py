import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


class CRNDataset(Dataset):
    def __init__(
        self, max_input_height_width: tuple, root: str, split: str, num_classes: int
    ):
        super(CRNDataset, self).__init__()
        self.num_classes = num_classes
        self.dataset: Cityscapes = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )

        self.max_input_height_width = max_input_height_width

        self.image_resize_transform = transforms.Compose(
            [
                transforms.Resize(max_input_height_width, Image.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    max_input_height_width,
                    Image.NEAREST,  # NEAREST as the values are categories and are not continuous
                ),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).long()[0]),
                transforms.Lambda(lambda x: one_hot(x, self.num_classes)),
                transforms.Lambda(lambda x: x.transpose(0, 2).transpose(1, 2)),
                transforms.Lambda(lambda x: x.float()),
            ]
        )

    def __getitem__(self, index):
        img, msk = self.dataset.__getitem__(index)

        img = self.image_resize_transform(img)
        msk = self.mask_resize_transform(msk)
        return img, msk

    def __len__(self):
        return self.dataset.__len__()

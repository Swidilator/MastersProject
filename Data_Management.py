import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


class CRNDataset(Dataset):
    def __init__(
        self,
        max_input_height_width: tuple,
        root: str,
        split: str,
        num_classes: int
        # mode: str,
        # target_type: str
    ):
        super(CRNDataset, self).__init__()
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        self.num_classes = num_classes
        self.dataset: Cityscapes = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )
        self.image_resize_transforms: list = []
        self.mask_resize_transforms: list = []

        curr_size = 2
        while curr_size < max_input_height_width[1]:

            curr_size = curr_size * 2
            print(curr_size)
            self.image_resize_transforms.append(
                transforms.Compose(
                    [
                        transforms.Resize(
                            (int(curr_size / 2), curr_size), Image.BILINEAR
                        ),
                        transforms.ToTensor(),
                    ]
                )
            )
            self.mask_resize_transforms.append(
                transforms.Compose(
                    [
                        transforms.Resize(
                            (int(curr_size / 2), curr_size), Image.NEAREST # NEAREST as the values are categories and are not continuous
                        ),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: (x * 255).long()[0]),
                        transforms.Lambda(lambda x: one_hot(x, self.num_classes)),
                    ]
                )
            )

    def __getitem__(self, index):
        images = []
        masks = []
        img, msk = self.dataset.__getitem__(index)
        tens = transforms.ToTensor()
        for i in range(len(self.image_resize_transforms)):
            images.append(self.image_resize_transforms[i](img))
            masks.append(self.mask_resize_transforms[i](msk))
        return images, masks

    def __len__(self):
        return self.dataset.__len__()

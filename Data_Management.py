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
            root=root,
            split=split,
            mode="fine",
            target_type="semantic"
        )

        self.max_input_height_width = max_input_height_width

        self.image_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    max_input_height_width,
                    Image.BILINEAR
                ),
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

        # img_channels: int = 3
        # img_ten_len: int = len(self.image_resize_transforms) * img_channels
        # img_ten_size = (
        #     img_ten_len,
        #     self.max_input_height_width[0],
        #     self.max_input_height_width[1],
        # )
        # images = torch.zeros(img_ten_size)
        #
        # msk_ten_len: int = len(self.image_resize_transforms) * self.num_classes
        # msk_ten_size = (
        #     msk_ten_len,
        #     self.max_input_height_width[0],
        #     self.max_input_height_width[1],
        # )
        # masks = torch.zeros(msk_ten_size)
        #
        # img, msk = self.dataset.__getitem__(index)
        # for i in range(len(self.image_resize_transforms)):
        #     img_i = self.image_resize_transforms[i](img)
        #     msk_i = self.mask_resize_transforms[i](msk)
        #
        #     images[
        #     i * img_channels: (i + 1) * img_channels,
        #     : 2 ** (i + 2),
        #     : 2 ** (i + 3),
        #     ] = img_i
        #     masks[
        #     i * self.num_classes: (i + 1) * self.num_classes,
        #     : 2 ** (i + 2),
        #     : 2 ** (i + 3),
        #     ] = msk_i
        #
        # return images, masks

    def __len__(self):
        return self.dataset.__len__()

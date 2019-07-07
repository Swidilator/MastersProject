from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import torch


class CRNDataset(Dataset):
    def __init__(
        self,
        max_input_height_width: tuple,
        root: str,
        split: str,
        # mode: str,
        # target_type: str
    ):
        super(CRNDataset, self).__init__()
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        self.dataset: Cityscapes = Cityscapes(
            root=root, split=split, mode="fine", target_type="semantic"
        )
        self.transforms: list = []

        scale_lambda = lambda x: x * 255

        curr_size = 1
        while curr_size < max_input_height_width[1]:

            curr_size = curr_size * 2
            print(curr_size)
            self.transforms.append(
                transforms.Compose(
                    [
                        transforms.Resize((int(curr_size / 2), curr_size)),
                        transforms.ToTensor(),
                        transforms.Lambda(scale_lambda),
                    ]
                )
            )

    def __getitem__(self, index):
        images = []
        masks = []
        img, msk = self.dataset.__getitem__(index)
        tens = transforms.ToTensor()
        msk_tens = tens(msk)
        msk_tens_ids = (msk_tens * 255).int()
        for i in range(len(self.transforms)):
            images.append(self.transforms[i](img))
            masks.append(self.transforms[i](msk))

        size = masks[1].shape
        return images, masks

    def __len__(self):
        return self.dataset.__len__()


def onehot_encode_target(target: torch.Tensor, num_classes: int):
    one_hot_tensor = torch.zeros
    for i in range(num_classes):
        # TODO
        pass

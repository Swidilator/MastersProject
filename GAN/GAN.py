import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
from math import log2
import time
import random

from Helper_Stuff import *
from Data_Management import GANDataset
from Training_Framework import MastersModel
from GAN.Generator import *

import wandb


class GANFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        max_input_height_width: image_size,
        num_classes: int,
        batch_size: int,
        num_loader_workers: int,
    ):
        super(GANFramework, self).__init__(device=device)
        self.batch_size = batch_size

        self.model_name: str = "GAN_Latest.pt"

        self.__set_data_loader__(
            data_path,
            max_input_height_width,
            num_classes,
            batch_size,
            num_loader_workers,
        )

        self.__set_model__(num_classes)

    @property
    def wandb_trainable_model(self):
        pass

    def __set_data_loader__(
        self,
        data_path,
        max_input_height_width,
        num_classes,
        batch_size,
        num_loader_workers,
    ):
        self.__data_set_train__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="train",
            num_classes=num_classes,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_test__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="test",
            num_classes=num_classes,
        )

        self.data_loader_test: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_test__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_val__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="val",
            num_classes=num_classes,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

    def __set_model__(self, num_classes) -> None:

        self.generator: Generator = Generator(num_classes)
        self.generator = self.generator.to(self.device)
        pass

    def save_model(self, model_dir: str) -> None:
        pass

    def load_model(self, model_dir: str, model_name: str) -> None:
        pass

    def train(self) -> epoch_output:
        pass

    def eval(self) -> epoch_output:
        pass

    def sample(self, k: int) -> sample_output:
        self.generator.eval()
        sample_list: list = random.sample(range(len(self.__data_set_test__)), k)
        outputs: sample_output = []
        # noise: torch.Tensor = torch.randn(
        #     1,
        #     1,
        #     self.input_tensor_size[0],
        #     self.input_tensor_size[1],
        #     device=self.device,
        # )
        transform: transforms.ToPILImage = transforms.ToPILImage()
        for i, val in enumerate(sample_list):
            img, msk = self.__data_set_test__[val]
            msk = msk.to(self.device).unsqueeze(0)
            img_out: torch.Tensor = self.generator(inputs=(msk, False))
            img_out = img_out.cpu()[0]
            outputs.append(transform(img_out))
            del img, msk
        return outputs

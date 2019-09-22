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
from GAN.Discriminator import *

import wandb


class GANFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        batch_size: int,
        num_loader_workers: int,
        subset_size: int,
        settings: dict,
    ):
        super(GANFramework, self).__init__(
            device, data_path, batch_size, num_loader_workers, subset_size, settings
        )
        self.model_name: str = "GAN"

        max_input_height_width: tuple = settings["max_input_height_width"]
        num_classes: int = settings["num_classes"]

        num_discriminators: int = 3

        self.__set_data_loader__(
            data_path,
            batch_size,
            num_loader_workers,
            subset_size,
            settings={
                "max_input_height_width": max_input_height_width,
                "num_classes": num_classes,
            },
        )

        self.__set_model__(
            settings={
                "num_classes": num_classes,
                "num_discriminators": num_discriminators,
            }
        )

    # TODO Confirm WandB works on multiple models
    @property
    def wandb_trainable_model(self) -> tuple:
        return tuple([self.generator, self.discriminator])

    def __set_data_loader__(
        self, data_path, batch_size, num_loader_workers, subset_size, settings
    ):
        max_input_height_width = settings["max_input_height_width"]
        num_classes: int = settings["num_classes"]

        self.__data_set_train__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="train",
            num_classes=num_classes,
            should_flip=True,
            subset_size=subset_size,
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
            should_flip=False,
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
            should_flip=False,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

    def __set_model__(self, settings) -> None:

        num_classes = settings["num_classes"]
        num_discriminators = settings["num_discriminators"]

        self.generator: Generator = Generator(num_classes)
        self.generator = self.generator.to(self.device)

        self.discriminator: FullDiscriminator = FullDiscriminator(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.optimizer_D: torch.optim.Adam = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002,
            # betas=(0.9, 0.999),
            # eps=1e-08,
            # weight_decay=0,
        )

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0002,
            # betas=(0.9, 0.999),
            # eps=1e-08,
            # weight_decay=0,
        )

        # TODO Change to correct criterion
        self.criterion = nn.BCELoss()

    def save_model(self, model_dir: str, snapshot: bool = False) -> None:
        super().save_model(model_dir, snapshot)

        model_snapshot: str = self.__get_model_snapshot_name__()

        save_dict: dict = {
            "model_state_dict_gen": (self.generator.state_dict()),
            "model_state_dict_disc": self.discriminator.state_dict(),
        }

        if snapshot:
            torch.save(save_dict, model_dir + model_snapshot)
        torch.save(save_dict, model_dir + self.model_name + "_Latest.pt")

    def load_model(self, model_dir: str, model_snapshot: str = None) -> None:
        super().load_model(model_dir, model_snapshot)
        if model_snapshot is not None:
            checkpoint = torch.load(
                model_dir + model_snapshot, map_location=self.device
            )
            self.generator.load_state_dict(checkpoint["model_state_dict_gen"])
            self.discriminator.load_state_dict(checkpoint["model_state_dict_disc"])
        else:
            checkpoint = torch.load(
                model_dir + self.model_name + "_Latest.pt", map_location=self.device
            )
            self.generator.load_state_dict(checkpoint["model_state_dict_gen"])
            self.discriminator.load_state_dict(checkpoint["model_state_dict_disc"])

    def train(self, input) -> epoch_output:
        self.generator.train()
        self.discriminator.train()

        torch.cuda.empty_cache()

        real_label = 1
        fake_label = 0

        loss_total: float = 0.0
        loss_ave: float = 0.0

        for batch_idx, (real_img, msk) in enumerate(self.data_loader_train):
            this_batch_size: int = msk.shape[0]
            real_img: torch.Tensor = real_img.to(self.device)
            msk: torch.Tensor = msk.to(self.device)

            # Discriminator
            self.optimizer_D.zero_grad()

            # TRAIN WITH ALL REAL BATCH
            label = torch.full((this_batch_size,), real_label, device=self.device)
            output = self.discriminator(real_img).view(-1)
            error_d_real = self.criterion(output, label)
            error_d_real.backward()
            error_d_real_mean = output.mean().item()

            # TRAIN WITH ALL-FAKE BATCH
            fake_img: torch.Tensor = self.generator(inputs=(msk, False))
            label.fill_(fake_label)
            output = self.discriminator(fake_img.detach()).view(-1)
            error_d_fake = self.criterion(output, label)
            error_d_fake.backward()
            error_d_fake_mean = output.mean().item()

            error_d = error_d_real + error_d_fake
            self.optimizer_D.step()

            # Generator
            self.generator.zero_grad()
            label.fill_(real_label)
            output = self.discriminator(fake_img).view(-1)
            error_g = self.criterion(output, label)
            error_g.backward()
            error_g_mean = output.mean().item()
            self.optimizer_G.step()

            del error_d, error_d_fake, error_d_real, error_g, msk, fake_img, real_img

            loss_ave += error_d_fake_mean + error_d_real_mean + error_g_mean
            loss_total += error_d_fake_mean + error_d_real_mean + error_g_mean
            if batch_idx * self.batch_size % 120 == 112:
                print(
                    "Batch: {batch}\nLoss: {loss_val}".format(
                        batch=batch_idx, loss_val=loss_ave * this_batch_size
                    )
                )
                # WandB logging, if WandB disabled this should skip the logging without error
                no_except(wandb.log, {"Batch Loss": loss_ave * this_batch_size})
                loss_ave = 0.0
        del loss_ave
        return loss_total, None

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

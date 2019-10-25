import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
from math import log2
import time
import random

from Helper_Stuff import *
from GAN.GAN_Dataset import GANDataset
from Training_Framework import MastersModel
from GAN.Generator import *
from GAN.Discriminator import *

# from GAN.Loss import *

import wandb


class GANFramework(MastersModel):
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        batch_size_slice: int,
        batch_size_total: int,
        num_loader_workers: int,
        subset_size: int,
        use_tanh: bool,
        use_input_noise: bool,
        settings: dict,
    ):
        super(GANFramework, self).__init__(
            device,
            data_path,
            batch_size_slice,
            batch_size_total,
            num_loader_workers,
            subset_size,
            use_tanh,
            use_input_noise,
            settings,
        )
        self.model_name: str = "GAN"

        max_input_height_width: tuple = settings["max_input_height_width"]
        num_classes: int = settings["num_classes"]

        num_discriminators: int = 3

        self.__set_data_loader__(
            data_path,
            batch_size_total,
            num_loader_workers,
            subset_size,
            use_input_noise,
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
        self,
        data_path,
        batch_size_total,
        num_loader_workers,
        subset_size,
        use_input_noise,
        settings,
    ):
        max_input_height_width = settings["max_input_height_width"]
        num_classes: int = settings["num_classes"]

        if batch_size_total > 16:
            self.medium_batch_size: int = 16
        else:
            self.medium_batch_size: int = batch_size_total

        self.__data_set_train__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="train",
            num_classes=num_classes,
            should_flip=True,
            subset_size=subset_size,
            noise=use_input_noise,
        )

        self.data_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_train__,
            batch_size=self.medium_batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_test__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="test",
            num_classes=num_classes,
            should_flip=False,
            subset_size=0,
            noise=False,
        )

        self.data_loader_test: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_test__,
            batch_size=self.medium_batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

        self.__data_set_val__ = GANDataset(
            max_input_height_width=max_input_height_width,
            root=data_path,
            split="val",
            num_classes=num_classes,
            should_flip=False,
            subset_size=0,
            noise=False,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.__data_set_val__,
            batch_size=self.medium_batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
        )

    def __set_model__(self, settings) -> None:

        num_classes = settings["num_classes"]

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

        # self.criterion = nn.BCELoss()
        # self.criterion = LSLossSingle()
        self.criterion = nn.MSELoss()

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

    def train(self, input: tuple) -> epoch_output:
        """

        Args:
            input:

        Returns:

        """
        self.generator.train()
        self.discriminator.train()
        torch.cuda.empty_cache()

        use_noisy_labels: bool = False

        real_label: float = 1
        fake_label: float = 0

        real_label_noise: float = (torch.rand(1).item() * 0.4) + 0.8
        fake_label_noise: float = (torch.rand(1).item() * 0.2)

        loss_total: float = 0.0
        loss_ave: float = 0.0

        # Metrics
        output_d_real_mean: float = 0.0
        output_d_fake_mean: float = 0.0
        error_d_real_measure: float = 0.0
        error_d_fake_measure: float = 0.0
        error_d_measure: float = 0.0

        error_g_measure: float = 0.0
        output_g_mean: float = 0.0

        # Logic for big batch, whereby we have a large value for a batch, but dataloader provides medium batch
        # medium_batch_per_big_batch: int = int(
        #     self.batch_size_total / self.max_data_loader_batch_size
        # )

        # mini_batch_per_big_batch: int = int(
        #     self.batch_size_total / self.batch_size_slice
        # )

        # # Number of times the medium batch should be looped over, given the slice size
        # if self.batch_size_total > self.max_data_loader_batch_size:
        #     mini_batch_per_medium_batch: int = int(
        #         self.max_data_loader_batch_size / self.batch_size_slice
        #     )
        # else:
        #     mini_batch_per_medium_batch: int = mini_batch_per_big_batch

        # Number of times the medium batch should be looped over, given the slice size
        mini_batch_per_medium_batch: int = self.medium_batch_size // self.batch_size_slice

        current_big_batch: int = 0

        # if medium_batch_per_big_batch < 1:
        #     medium_batch_per_big_batch = 1

        # Initially, there have been no mini batches
        # current_medium_batch: int = 0

        # Increments as mini batches are processed, should be equal to big batch eventually
        this_big_batch_size: int = 0

        final_medium_batch: bool = False

        # Medium batch
        for batch_idx, (real_img_total, msk_total) in enumerate(self.data_loader_train):
            this_medium_batch_size: int = real_img_total.shape[0]

            if (this_medium_batch_size < self.medium_batch_size) or (
                (batch_idx + 1) * self.medium_batch_size
                == len(self.data_loader_train.dataset)
            ):
                final_medium_batch = True

            # Keep track of number of small batches
            # current_medium_batch += 1

            # # Prevent a batch of zero from being run
            # if current_medium_batch % medium_batch_per_big_batch == 0:
            #     final_mini_batch = True
            #     #this_big_batch_size = 0

            # Discriminator
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()

            # Loop over medium batch
            for i in range(mini_batch_per_medium_batch):

                # May result in a mini batch size of 0
                real_img: torch.Tensor = real_img_total[
                    i * self.batch_size_slice : (i + 1) * self.batch_size_slice
                ].to(self.device)
                msk: torch.Tensor = msk_total[
                    i * self.batch_size_slice : (i + 1) * self.batch_size_slice
                ].to(self.device)

                this_mini_batch_size: int = msk.shape[0]

                if (
                    this_mini_batch_size == 0
                ):  # Empty mini batch, medium batch is last in epoch
                    final_mini_batch = True
                    break
                elif this_mini_batch_size != self.batch_size_slice:
                    final_mini_batch = True

                # Add the mini batch size to the big batch size
                # ONLY DO THIS ONCE
                this_big_batch_size += this_mini_batch_size
                # print(this_big_batch_size)

                # TRAIN WITH ALL REAL BATCH
                output = self.discriminator(real_img)
                with torch.no_grad():
                    if not use_noisy_labels:
                        label = torch.full(
                            (output.shape[0],), real_label, device=self.device
                        )
                    else:
                        label = torch.full(
                            (output.shape[0],), real_label_noise, device=self.device
                        )
                error_d_real: torch.Tensor = self.criterion(output, label)
                error_d_real.backward()
                error_d_real_measure += error_d_real.item()
                output_d_real_mean += output.mean().item()

                # TRAIN WITH ALL-FAKE BATCH
                fake_img: torch.Tensor = self.generator(inputs=(msk, False))[0]
                if not use_noisy_labels:
                    label.fill_(fake_label)
                else:
                    label.fill_(fake_label_noise)
                output: torch.Tensor = self.discriminator(fake_img.detach())
                error_d_fake: torch.Tensor = self.criterion(output, label)
                error_d_fake.backward()
                error_d_fake_measure += error_d_fake.item()
                output_d_fake_mean += output.mean().item()

                error_d: torch.Tensor = error_d_real + error_d_fake
                error_d_measure += error_d.item()

                loss_ave += error_d.item()
                loss_total += error_d.item()

                del output, fake_img, error_d_fake, error_d_real, error_d

            # Perform update for discriminator
            # if current_medium_batch % medium_batch_per_big_batch == 0:
            if (this_big_batch_size == self.batch_size_total) or final_medium_batch:
                i: torch.nn.Parameter
                for i in self.discriminator.parameters():
                    i.grad = i.grad / (self.batch_size_total / self.batch_size_slice)
                # print("Stepping")
                self.optimizer_D.step()

            # Generator
            # Loop over medium batch
            for i in range(mini_batch_per_medium_batch):
                msk: torch.Tensor = msk_total[
                    i * self.batch_size_slice : (i + 1) * self.batch_size_slice
                ].to(self.device)
                fake_img: torch.Tensor = self.generator(inputs=(msk, False))[0]

                this_mini_batch_size: int = msk.shape[0]

                if (
                    this_mini_batch_size == 0
                ):  # Empty mini batch, medium batch is last in epoch
                    # final_mini_batch = True
                    break
                elif (
                    this_mini_batch_size != self.batch_size_slice
                ):  # Non-full mini batch, medium batch is last in epoch
                    # final_mini_batch = True
                    pass

                if not use_noisy_labels:
                    label.fill_(real_label)
                else:
                    label.fill_(real_label_noise)
                output: torch.Tensor = self.discriminator(fake_img)
                error_g: torch.Tensor = self.criterion(output, label)
                error_g.backward()
                error_g_measure += error_g.item()
                output_g_mean += output.mean().item()

                loss_ave += error_g.item()
                loss_total += error_g.item()

                del output, fake_img, error_g

            # Perform update for generator
            if (this_big_batch_size == self.batch_size_total) or final_medium_batch:
                #print(current_big_batch)
                i: torch.nn.Parameter
                for i in self.generator.parameters():
                    if i.grad is not None:
                        i.grad = i.grad / (
                            self.batch_size_total / self.batch_size_slice
                        )
                # print("Stepping")
                self.optimizer_G.step()

            # Final check, update/reset all values
            if (this_big_batch_size == self.batch_size_total) or final_medium_batch:
                # Step big batch count
                current_big_batch += 1

                # Normalise this accumulated error
                output_scaling_factor: float = (this_big_batch_size / self.batch_size_slice)
                print(output_scaling_factor)
                output_d_real_mean = output_d_real_mean / output_scaling_factor
                output_d_fake_mean = output_d_fake_mean / output_scaling_factor
                # error_d_real_measure = error_d_real_measure / output_scaling_factor
                # error_d_fake_measure = error_d_fake_measure / output_scaling_factor
                error_d_measure = error_d_measure / output_scaling_factor

                error_g_measure = error_g_measure / output_scaling_factor
                output_g_mean = output_g_mean / output_scaling_factor

                # TODO Add correct and useful WandB logging
                print(
                    """
                    Batch:          {batch}
                    Loss:           {loss_val}
                    Error D:        {error_d}
                    Error G:        {error_g}
                    Error G Mean:   {error_g_mean}
                    Output D Fake:  {output_d_fake_mean}
                    Output D Real:  {output_d_real_mean}""".format(
                        batch=current_big_batch,
                        loss_val=(loss_ave / (current_big_batch / self.batch_size_slice)),
                        error_d=error_d_measure,
                        error_g=error_g_measure,
                        error_g_mean=output_g_mean,
                        output_d_fake_mean=output_d_fake_mean,
                        output_d_real_mean=output_d_real_mean,
                    )
                )

                # TODO Check this is completed
                if (
                    (self.batch_size_total > 8)
                    or (current_big_batch * self.batch_size_total % 120 >= 112)
                    or final_medium_batch
                ):
                    batch_loss_val: float = (
                        loss_ave / this_big_batch_size
                    ) * self.batch_size_total

                    print(
                        "Batch: {batch}\nLoss: {loss_val}".format(
                            batch=current_big_batch, loss_val=batch_loss_val
                        )
                    )
                    # WandB logging, if WandB disabled this should skip the logging without error
                    no_except(wandb.log, {"Batch Loss": batch_loss_val})
                    loss_ave = 0.0

                # Reset big batch metrics
                this_big_batch_size = 0
                output_d_real_mean = 0.0
                output_d_fake_mean = 0.0
                output_g_mean = 0.0
                error_d_fake_measure = 0.0
                error_g_measure = 0.0
                error_d_real_measure = 0.0
                error_d_measure = 0.0

            del msk_total, real_img_total
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
            img_out = img_out[1][0].cpu()
            outputs.append(transform(img_out))
            del img, msk
        return outputs

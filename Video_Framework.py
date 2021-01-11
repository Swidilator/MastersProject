import os
from contextlib import nullcontext
from itertools import chain
from typing import Any, Tuple, List, Union, Optional

import torch
from torch.cuda import amp as torch_amp
import wandb
from PIL import ImageFile
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from CRN import CRNVideo
from GAN import Generator
from support_scripts.components import (
    FeatureEncoder,
    FlowNetWrapper,
    FullDiscriminator,
    feature_matching_error,
    PerceptualLossNetwork,
)
from support_scripts.sampling import SampleDataHolder
from support_scripts.utils import (
    MastersModel,
    ModelSettingsManager,
    CityScapesVideoDataset,
    CityScapesStandardDataset,
    collate_fn,
)

import flowiz as fz

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoFramework(MastersModel):
    def __init__(
        self,
        model: str,
        device: torch.device,
        dataset_path: str,
        input_image_height_width: tuple,
        batch_size: int,
        num_data_workers: int,
        training_subset_size: int,
        flip_training_images: bool,
        sample_only: bool,
        use_amp: Union[str, bool],
        log_every_n_steps: int,
        model_save_dir: str,
        image_save_dir: str,
        starting_epoch: int,
        num_frames_per_training_video: int,
        num_frames_per_sampling_video: int,
        prior_frame_seed_type: str,
        use_mask_for_instances: bool,
        video_frame_offset: Union[int, str],
        use_saved_feature_encodings: bool,
        **kwargs,
    ):
        super(VideoFramework, self).__init__(
            model,
            device,
            dataset_path,
            input_image_height_width,
            batch_size,
            num_data_workers,
            training_subset_size,
            flip_training_images,
            sample_only,
            use_amp,
            log_every_n_steps,
            model_save_dir,
            image_save_dir,
            starting_epoch,
            num_frames_per_training_video,
            num_frames_per_sampling_video,
            prior_frame_seed_type,
            use_mask_for_instances,
            video_frame_offset,
            use_saved_feature_encodings,
            **kwargs,
        )

        # fmt: off
        self.base_learning_rate: float = kwargs["base_learning_rate"]
        self.decay_learning_rate: bool = kwargs["decay_learning_rate"]
        self.decay_staring_epoch: int = kwargs["decay_staring_epoch"]
        self.use_noisy_labels: bool = kwargs["use_noisy_labels"]
        self.use_feature_encodings: bool = kwargs["use_feature_encodings"]
        self.use_edge_map: bool = kwargs["use_edge_map"]
        self.use_perceptual_loss: bool = kwargs["use_perceptual_loss"]
        self.perceptual_base_model: str = kwargs["perceptual_base_model"]
        self.use_perceptual_loss_output_image: bool = kwargs["use_perceptual_loss_output_image"]
        self.perceptual_loss_scaling_method: str = kwargs["perceptual_loss_scaling_method"]
        self.num_discriminators: int = kwargs["num_discriminators"]
        self.feature_matching_weight: float = kwargs["feature_matching_weight"]
        self.use_sigmoid_discriminator: bool = kwargs["use_sigmoid_discriminator"]
        self.use_twin_network: bool = kwargs["use_twin_network"]
        self.use_optical_flow: bool = kwargs["use_optical_flow"]
        self.use_tanh: bool = kwargs["use_tanh"]
        self.num_prior_frames: int = kwargs["num_prior_frames"]
        self.normalise_prior_frames: int = kwargs["normalise_prior_frames"]

        self.use_local_enhancer: bool = kwargs["use_local_enhancer"]
        self.lock_global_generator: bool = kwargs["lock_global_generator"]
        self.unlock_global_generator_epoch: bool = kwargs["unlock_global_generator_epoch"]

        self.num_output_images: int = kwargs["num_output_images"]
        self.input_tensor_size: tuple = kwargs["input_tensor_size"]
        self.num_inner_channels: int = kwargs["num_inner_channels"]
        self.layer_norm_type: str = kwargs["layer_norm_type"]
        self.use_resnet_rms: bool = kwargs["use_resnet_rms"]
        self.num_resnet_processing_rms: int = kwargs["num_resnet_processing_rms"]

        self.flownet_save_path: str = kwargs["flownet_save_path"]  # This is in args
        # fmt: on

        self.__set_data_loader__()

        self.__set_model__()

    @property
    def data_set_train(self) -> torch.utils.data.Dataset:
        return self.dataset_train

    @property
    def data_set_val(self) -> torch.utils.data.Dataset:
        return self.dataset_val

    @property
    def wandb_trainable_model(self) -> tuple:
        models = [self.generator]
        if self.use_feature_encodings:
            models.append(self.feature_encoder)
        if self.num_discriminators > 0:
            models.append(self.image_discriminator)
            if self.use_optical_flow:
                models.append(self.flow_discriminator)
        return tuple(models)

    @classmethod
    def from_model_settings_manager(
        cls, manager: ModelSettingsManager
    ) -> "VideoFramework":
        """
        Create a GANFramework instance from a ModelSettingsManager instance
        instead of manually inputting arguments.

        :param manager: ModelSettingsManager to use.
        :return: GANFramework instantiated from manager
        """

        # fmt: off
        settings = {
            "base_learning_rate": manager.model_conf["BASE_LEARNING_RATE"],
            "decay_learning_rate": manager.model_conf["DECAY_LEARNING_RATE"],
            "decay_staring_epoch": manager.model_conf["DECAY_STARTING_EPOCH"],
            "use_noisy_labels": manager.model_conf["USE_NOISY_LABELS"],
            "use_feature_encodings": manager.model_conf["USE_FEATURE_ENCODINGS"],
            "use_edge_map": manager.model_conf["USE_EDGE_MAP"],
            "use_perceptual_loss": manager.model_conf["USE_PERCEPTUAL_LOSS"],
            "perceptual_base_model": manager.model_conf["PERCEPTUAL_BASE_MODEL"],
            "use_perceptual_loss_output_image": manager.model_conf["USE_PERCEPTUAL_LOSS_OUTPUT_IMAGE"],
            "perceptual_loss_scaling_method": manager.model_conf["PERCEPTUAL_LOSS_SCALING_METHOD"],
            "num_discriminators": manager.model_conf["NUM_DISCRIMINATORS"],
            "feature_matching_weight": manager.model_conf["FEATURE_MATCHING_WEIGHT"],
            "use_sigmoid_discriminator": manager.model_conf["USE_SIGMOID_DISCRIMINATOR"],
            "use_twin_network": manager.model_conf["USE_TWIN_NETWORK"],
            "use_optical_flow": manager.model_conf["USE_OPTICAL_FLOW"],
            "use_tanh": manager.model_conf["USE_TANH"],
            "num_prior_frames": manager.model_conf["NUM_PRIOR_FRAMES"],
            "normalise_prior_frames": manager.model_conf["NORMALISE_PRIOR_FRAMES"],

            "use_local_enhancer": manager.model_conf["GAN_USE_LOCAL_ENHANCER"],
            "lock_global_generator": manager.model_conf["GAN_LOCK_GLOBAL_GENERATOR"],
            "unlock_global_generator_epoch": manager.model_conf["GAN_UNLOCK_GLOBAL_GENERATOR_EPOCH"],

            "num_output_images": manager.model_conf["CRN_NUM_OUTPUT_IMAGES"],
            "input_tensor_size": (
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_HEIGHT"],
                manager.model_conf["CRN_INPUT_TENSOR_SIZE_WIDTH"],
            ),
            "num_inner_channels": manager.model_conf["CRN_NUM_INNER_CHANNELS"],
            "layer_norm_type": manager.model_conf["CRN_LAYER_NORM_TYPE"],
            "use_resnet_rms": manager.model_conf["CRN_USE_RESNET_RMS"],
            "num_resnet_processing_rms": manager.model_conf["CRN_NUM_RESNET_PROCESSING_RMS"],
        }
        # fmt: on

        return cls(**manager.args, **settings)

    def __set_data_loader__(self, **kwargs):

        if self.use_optical_flow:
            assert (
                self.num_frames_per_sampling_video > 1
            ), "self.use_optical_flow is True, but self.num_frames_per_sampling_video == 1."
            if not self.sample_only:
                assert (
                    self.num_frames_per_training_video > 1
                ), "self.use_optical_flow is True, but self.num_frames_per_training_video == 1."

        skip_first_training_frame: int = (
            self.prior_frame_seed_type == "real" or self.use_optical_flow
        ) and not self.sample_only
        num_frames_per_training_video = (
            self.num_frames_per_training_video + skip_first_training_frame
        )
        num_frames_per_sampling_video = (
            self.num_frames_per_sampling_video
            + self.num_prior_frames
            + self.use_optical_flow
        )

        if num_frames_per_training_video > 1:
            dataset_train = CityScapesVideoDataset
            root = self.dataset_path + "/sequence"
            generated_data: bool = True
        else:
            dataset_train = CityScapesStandardDataset
            root = self.dataset_path
            generated_data: bool = False

        self.dataset_train = dataset_train(
            root=root,
            split="train",
            should_flip=self.flip_training_images,
            subset_size=self.training_subset_size,
            output_image_height_width=self.input_image_height_width,
            generated_data=generated_data,
            num_frames=num_frames_per_training_video,
            frame_offset="random",
        )

        self.data_loader_train: torch.utils.data.DataLoader = (
            torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_data_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        )

        self.dataset_val = CityScapesVideoDataset(
            root=self.dataset_path + "/sequence",
            split="val",
            should_flip=False,
            subset_size=0,
            output_image_height_width=self.input_image_height_width,
            generated_data=True,
            num_frames=num_frames_per_sampling_video,
            frame_offset=self.video_frame_offset,
        )

        self.data_loader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        self.dataset_demoVideo = CityScapesStandardDataset(
            root=self.dataset_path,
            split="demoVideo",
            should_flip=False,
            subset_size=0,
            output_image_height_width=self.input_image_height_width,
            generated_data=True,
            num_frames=1,
            frame_offset="random",
        )

        self.num_classes = self.dataset_train.num_output_segmentation_classes

    def __set_model__(self, **kwargs) -> None:
        # Useful channel count variables
        num_image_channels: int = 3
        num_edge_map_channels: int = self.use_edge_map * 1
        num_feature_encoding_channels: int = self.use_feature_encodings * 3
        num_flow_channels: int = 2

        use_mask_for_instances: bool = True

        # Feature Encoder
        if self.use_feature_encodings:
            self.feature_encoder: FeatureEncoder = FeatureEncoder(
                num_image_channels,
                num_feature_encoding_channels,
                4,
                self.device,
                self.model_save_dir,
                self.use_saved_feature_encodings,
                use_mask_for_instances,
                self.num_classes,
            )
            self.feature_encoder = self.feature_encoder.to(self.device)
            if self.use_saved_feature_encodings:
                self.feature_encoder.eval()
                for param in self.feature_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.feature_encoder.parameters():
                    param.requires_grad = True

        # Generator network
        if self.model == "GAN":
            # Generator
            self.generator: Generator = Generator(
                self.use_tanh,
                self.num_classes,
                self.use_feature_encodings,
                self.num_prior_frames,
                self.use_optical_flow,
                self.use_edge_map,
                self.use_twin_network,
                self.use_local_enhancer,
                self.input_image_height_width,
            )
        elif self.model == "CRN":
            self.generator: CRNVideo = CRNVideo(
                use_tanh=self.use_tanh,
                input_tensor_size=self.input_tensor_size,
                final_image_size=self.input_image_height_width,
                num_classes=self.num_classes,
                num_inner_channels=self.num_inner_channels,
                use_feature_encoder=self.use_feature_encodings,
                layer_norm_type=self.layer_norm_type,
                use_resnet_rms=self.use_resnet_rms,
                num_resnet_processing_rms=self.num_resnet_processing_rms,
                num_prior_frames=self.num_prior_frames,
                use_optical_flow=self.use_optical_flow,
                use_edge_map=self.use_edge_map,
                use_twin_network=self.use_twin_network,
                num_output_images=self.num_output_images,
            )
        print(self.generator)
        self.generator = self.generator.to(self.device)

        self.crn_video = self.generator
        self.crn = self.generator

        if not self.sample_only:

            # Create params depending on what needs to be trained
            params = (
                self.generator.local_enhancer.parameters()
                if self.use_local_enhancer
                else []
            )
            if self.model == "GAN":
                if (
                    not self.lock_global_generator
                    or self.unlock_global_generator_epoch <= self.starting_epoch
                ):
                    params = chain(
                        params,
                        self.generator.global_generator.parameters(),
                    )
                    if self.use_feature_encodings and (
                        not self.use_saved_feature_encodings
                    ):
                        params = chain(
                            params,
                            self.feature_encoder.parameters(),
                        )
                self.optimizer_G = torch.optim.Adam(
                    params,
                    lr=self.base_learning_rate,
                    betas=(0.5, 0.999),
                    # eps=1e-08,
                    # weight_decay=0,
                )
            elif self.model == "CRN":
                params = chain(
                    params,
                    self.generator.parameters(),
                )
                if self.use_feature_encodings and (
                    not self.use_saved_feature_encodings
                ):
                    params = chain(
                        params,
                        self.feature_encoder.parameters(),
                    )
                self.optimizer_G = torch.optim.Adam(
                    params,
                    lr=self.base_learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=0,
                )

            # Perceptual Loss
            if self.use_perceptual_loss:
                self.loss_net: PerceptualLossNetwork = PerceptualLossNetwork(
                    self.perceptual_base_model,
                    self.device,
                    self.use_perceptual_loss_output_image,
                    self.perceptual_loss_scaling_method,
                )
                self.loss_net = self.loss_net.to(self.device)

            # Flownet for video training
            if self.use_optical_flow:
                assert (
                    self.num_prior_frames > 0
                ), "self.use_optical_flow == True, but self.num_prior_frames == 0."
                assert (
                    self.num_frames_per_training_video > 0
                ), "self.use_optical_flow == True, but self.num_frames_per_training_video == 0."

                self.flow_criterion = torch.nn.L1Loss()

                self.flownet = FlowNetWrapper(self.flownet_save_path)
                self.flownet = self.flownet.to(self.device)

            if self.num_discriminators > 0:
                # Discriminator criterion
                self.criterion_D = nn.MSELoss()

                # Image discriminator
                self.image_discriminator_input_channel_count: int = (
                    self.num_classes
                    + num_edge_map_channels
                    + num_image_channels
                    + (self.num_prior_frames * (self.num_classes + num_image_channels))
                )

                self.image_discriminator: FullDiscriminator = FullDiscriminator(
                    self.device,
                    self.image_discriminator_input_channel_count,
                    self.num_discriminators,
                    self.use_sigmoid_discriminator,
                )
                self.image_discriminator = self.image_discriminator.to(self.device)

                self.optimizer_D_image: torch.optim.Adam = torch.optim.Adam(
                    self.image_discriminator.parameters(),
                    lr=0.0001,
                    betas=(0.5, 0.999),
                    # eps=1e-08,
                    # weight_decay=0,
                )

                # Flownet for video training
                if self.use_optical_flow:
                    self.flow_discriminator_input_channel_count: int = (
                        self.num_classes
                        + num_edge_map_channels
                        + num_flow_channels
                        + (self.num_prior_frames * self.num_classes)
                        + ((self.num_prior_frames - 1) * num_flow_channels)
                    )

                    self.flow_discriminator: FullDiscriminator = FullDiscriminator(
                        self.device,
                        self.flow_discriminator_input_channel_count,
                        self.num_discriminators,
                        self.use_sigmoid_discriminator,
                    )
                    self.flow_discriminator = self.flow_discriminator.to(self.device)

                    self.optimizer_D_flow: torch.optim.Adam = torch.optim.Adam(
                        self.flow_discriminator.parameters(),
                        lr=0.0001,
                        betas=(0.5, 0.999),
                        # eps=1e-08,
                        # weight_decay=0,
                    )

            self.torch_gradient_scaler: torch_amp.GradScaler = (
                torch_amp.GradScaler(enabled=self.use_amp == "torch")
            )

    def save_model(self, epoch: int = -1) -> None:
        super().save_model()

        if self.sample_only:
            raise RuntimeError("Cannot save model in 'sample_only' mode.")

        self.generator.zero_grad()
        if self.num_discriminators > 0:
            self.image_discriminator.zero_grad()
            if self.use_feature_encodings:
                self.feature_encoder.zero_grad()

        update_logs: dict = {"reordered_discriminators": True}

        save_dict: dict = {
            "model": self.model,
            "kwargs": self.kwargs,
            "update_logs": update_logs,
        }
        if self.model == "GAN":
            save_dict.update(
                {
                    "dict_global_generator": self.generator.global_generator.state_dict(),
                }
            )
            if self.use_local_enhancer:
                save_dict.update(
                    {"dict_local_enhancer": self.generator.local_enhancer.state_dict()}
                )
        elif self.model == "CRN":
            save_dict.update(
                {
                    "dict_crn": self.generator.state_dict(),
                }
            )
        else:
            raise ValueError("Invalid self.model.")

        if self.use_feature_encodings:
            save_dict.update(
                {"dict_encoder_decoder": self.feature_encoder.state_dict()}
            )

        # Save each image_discriminator individually
        for i in range(self.num_discriminators):
            save_dict.update(
                {
                    "dict_discriminator_{num}".format(
                        num=i
                    ): self.image_discriminator.discriminators[i].state_dict()
                }
            )
            if self.use_optical_flow:
                save_dict.update(
                    {
                        "dict_flow_discriminator_{num}".format(
                            num=i
                        ): self.flow_discriminator.discriminators[i].state_dict()
                    }
                )

        # Save optimisers
        save_dict.update({"optimizer_G": self.optimizer_G.state_dict()})
        if self.num_discriminators > 0:
            save_dict.update({"optimizer_D_image": self.optimizer_D_image.state_dict()})
            if self.use_optical_flow:
                save_dict.update(
                    {"optimizer_D_flow": self.optimizer_D_flow.state_dict()}
                )

        if epoch >= 0:
            epoch_file_name: str = os.path.join(
                self.model_save_dir,
                self.model_name + "_Epoch_{epoch}.pt".format(epoch=epoch),
            )
            torch.save(save_dict, epoch_file_name)

        latest_file_name: str = os.path.join(
            self.model_save_dir, self.model_name + "_Latest.pt"
        )
        torch.save(save_dict, latest_file_name)

    def load_model(self, model_file_name: str) -> None:
        super().load_model(model_file_name)

        # Create final model file path and output
        load_path: str = os.path.join(self.model_save_dir, model_file_name)
        print("Loading model:")
        print(load_path)

        checkpoint = torch.load(load_path, map_location=self.device)

        if "model" in checkpoint:
            assert (
                checkpoint["model"] == self.model
            ), "checkpoint['model'] does not match self.model."

        # Modifying older saves to newer standards
        update_logs: dict = {"reordered_discriminators": False}

        if "update_logs" in checkpoint:
            update_logs.update(checkpoint["update_logs"])

        if self.model == "GAN":
            self.generator.global_generator.load_state_dict(
                checkpoint["dict_global_generator"]
            )
            if self.use_local_enhancer and "dict_local_enhancer" in checkpoint:
                self.generator.local_enhancer.load_state_dict(
                    checkpoint["dict_local_enhancer"]
                )
            elif self.use_local_enhancer:
                print("Warning: No local enhancer found in model save file.")

        elif self.model == "CRN":
            self.generator.load_state_dict(checkpoint["dict_crn"], strict=False)

        if self.use_feature_encodings:
            self.feature_encoder.load_state_dict(checkpoint["dict_encoder_decoder"])

        if not self.sample_only:
            if self.num_discriminators > 0:
                if "dict_discriminator" in checkpoint:
                    self.image_discriminator.load_state_dict(
                        checkpoint["dict_discriminator"], strict=False
                    )
                else:
                    for i in range(self.num_discriminators):
                        if "dict_discriminator_{num}".format(num=i) in checkpoint:
                            self.image_discriminator.discriminators[i].load_state_dict(
                                checkpoint["dict_discriminator_{num}".format(num=i)]
                            )

                if self.use_optical_flow:
                    if "dict_flow_discriminator" in checkpoint:
                        self.flow_discriminator.load_state_dict(
                            checkpoint["dict_flow_discriminator"], strict=False
                        )
                    else:
                        for i in range(self.num_discriminators):
                            if (
                                "dict_flow_discriminator_{num}".format(num=i)
                                in checkpoint
                            ):
                                self.flow_discriminator.discriminators[
                                    i
                                ].load_state_dict(
                                    checkpoint[
                                        "dict_flow_discriminator_{num}".format(num=i)
                                    ]
                                )

                if update_logs["reordered_discriminators"] is False:
                    loaded_discriminators: list = [
                        int(key[-1])
                        for key, value in checkpoint.items()
                        if "image_discriminator" in key.lower()
                    ]
                    num_loaded_discriminators: int = len(loaded_discriminators)
                    self.image_discriminator.discriminators = nn.ModuleList(
                        [
                            *[
                                self.image_discriminator.discriminators[i]
                                for i in reversed(loaded_discriminators)
                            ],
                            *self.image_discriminator.discriminators[
                                num_loaded_discriminators:
                            ],
                        ]
                    )

            # Optimisers, not my proudest code
            try:
                if "optimizer_G" in checkpoint:
                    self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            except Exception:
                print("optimizer_G could not load state dict")
            if self.num_discriminators > 0:
                try:
                    if "optimizer_D_image" in checkpoint:
                        self.optimizer_D_image.load_state_dict(
                            checkpoint["optimizer_D_image"]
                        )
                except Exception:
                    print("optimizer_D_image could not load state dict")
                if self.use_optical_flow:
                    try:
                        if "optimizer_D_flow" in checkpoint:
                            self.optimizer_D_flow.load_state_dict(
                                checkpoint["optimizer_D_flow"]
                            )
                    except Exception:
                        print("optimizer_D_flow could not load state dict")

    @classmethod
    def load_model_with_embedded_settings(cls, manager: ModelSettingsManager):
        load_path: str = os.path.join(
            manager.args["model_save_dir"], manager.args["load_saved_model"]
        )
        checkpoint = torch.load(load_path)
        kwargs: dict = checkpoint["kwargs"]

        shared_keys: list = [x for x in manager.args.keys() if x in kwargs]
        for key in shared_keys:
            del kwargs[key]

        model_frame: VideoFramework = cls(**manager.args, **kwargs)
        model_frame.load_model(manager.args["load_saved_model"])
        return model_frame

    def train(self, **kwargs) -> Tuple[float, Any]:
        # Set all modules to train mode
        self.generator.train()
        if self.num_discriminators > 0:
            self.image_discriminator.train()
            if self.use_optical_flow:
                self.flow_discriminator.train()

        if not self.use_saved_feature_encodings and self.use_feature_encodings:
            self.feature_encoder.train()

        # If sampling from saved feature encodings, and using a single set of settings
        #  all sampling for a single epoch will have the same settings, but refresh between epochs
        if self.use_feature_encodings and self.use_saved_feature_encodings:
            self.feature_encoder.feature_extractions_sampler.update_single_setting_class_list()

        # Loss that will be returned after the epoch is complete
        loss_total: float = 0.0

        # Discriminator labels
        real_label: float = 1.0
        fake_label: float = 0.0
        real_label_gan: float = 1.0

        # Loop through dataset
        for batch_idx, input_dict in enumerate(
            tqdm(self.data_loader_train, desc="Training")
        ):
            # Batch size can be irregular at end of epoch, get this so always accurate
            this_batch_size: int = input_dict["img"].shape[0]
            num_frames: int = input_dict["img"].shape[1]

            log_this_batch: bool = (batch_idx % self.log_every_n_steps == 0) or (
                batch_idx == (len(self.data_loader_train) - 1)
            )

            if this_batch_size == 0:
                break

            # Set up prior frame lists with zeros filling the correct number of elements
            prior_fake_image_list: list = [
                torch.zeros_like(input_dict["img"][:, 0], device=self.device)
            ] * self.num_prior_frames
            prior_reference_image_list: list = [
                torch.zeros_like(input_dict["img"][:, 0], device=self.device)
            ] * self.num_prior_frames
            prior_mask_list: list = [
                torch.zeros_like(input_dict["msk"][:, 0], device=self.device)
            ] * self.num_prior_frames
            prior_real_flow_list: list = [
                torch.zeros_like(input_dict["img"][:, 0, 0:2], device=self.device)
            ] * (self.num_prior_frames - 1)
            prior_fake_flow_list: list = [
                torch.zeros_like(input_dict["img"][:, 0, 0:2], device=self.device)
            ] * (self.num_prior_frames - 1)

            # If a real image is required as a prior frame seed, populate that element
            if self.num_prior_frames > 0:
                if self.prior_frame_seed_type == "real":
                    prior_reference_image_list[0] = input_dict["img"][:, 0].to(
                        self.device
                    )
                    prior_fake_image_list[0] = input_dict["img"][:, 0].to(self.device)
                    prior_mask_list[0] = input_dict["msk"][:, 0].to(self.device)

            # Loss holders that may or may not be used later on
            video_loss: float = 0.0
            video_loss_img: float = 0.0
            video_loss_img_h: float = 0.0
            video_loss_flow: float = 0.0
            video_loss_warp: float = 0.0

            video_loss_d: float = 0.0
            video_loss_g_gan: float = 0.0
            video_loss_g_fm: float = 0.0
            video_output_d_real_mean: float = 0.0
            video_output_d_fake_mean: float = 0.0

            video_loss_d_flow: float = 0.0
            video_loss_g_gan_flow: float = 0.0
            video_loss_g_fm_flow: float = 0.0
            video_output_d_real_mean_flow: float = 0.0
            video_output_d_fake_mean_flow: float = 0.0

            # If using a method that needs info from a first frame, shift training by a frame to make sure things are
            # consistent
            skip_first_frame: int = (
                self.prior_frame_seed_type == "real" or self.use_optical_flow
            )

            num_frames_per_training_video = (
                self.num_frames_per_training_video + skip_first_frame
            )

            # Loop through each frame
            for frame_no in range(skip_first_frame, num_frames_per_training_video):
                # Zero any gradients, not explicitly necessary
                self.generator.zero_grad()
                if self.num_discriminators > 0:
                    self.image_discriminator.zero_grad()
                    if self.use_optical_flow:
                        self.flow_discriminator.zero_grad()

                # Get the single frame that will be used this loop
                reference_image: torch.Tensor = input_dict["img"][:, frame_no].to(
                    self.device
                )
                mask: torch.Tensor = input_dict["msk"][:, frame_no].to(self.device)
                instance: torch.Tensor = input_dict["inst"][:, frame_no].to(self.device)
                edge_map: torch.Tensor = input_dict["edge_map"][:, frame_no].to(
                    self.device
                )

                # Autocast if using amp
                with torch_amp.autocast(enabled=self.use_amp == "torch"):
                    # Perceptual loss
                    loss_img: torch.Tensor = torch.zeros(1, device=self.device)
                    # Image discriminator loss
                    loss_d_image: torch.Tensor = torch.zeros(1, device=self.device)
                    loss_g_image: torch.Tensor = torch.zeros(1, device=self.device)
                    # Flow related losses
                    loss_d_flow: torch.Tensor = torch.zeros(1, device=self.device)
                    loss_g_flow: torch.Tensor = torch.zeros(1, device=self.device)
                    loss_img_h: torch.Tensor = torch.zeros(1, device=self.device)
                    loss_flow: torch.Tensor = torch.zeros(1, device=self.device)
                    loss_warp: torch.Tensor = torch.zeros(1, device=self.device)

                    # Generate feature encoding for this frame if requested
                    if self.use_feature_encodings:
                        feature_encoding: Optional[torch.Tensor] = self.feature_encoder(
                            reference_image,
                            instance,
                            input_dict["img_id"]
                            if self.use_saved_feature_encodings
                            else None,
                            input_dict["img_flipped"],
                            mask,
                        )
                    else:
                        feature_encoding = None

                    # Setting output types
                    fake_img: torch.Tensor
                    fake_img_h: torch.Tensor
                    fake_img_w: torch.Tensor
                    fake_flow: torch.Tensor
                    fake_flow_mask: torch.Tensor
                    # Generate outputs from network
                    # fake_img has format (batch x num_images x num_channels x height x width)
                    (
                        fake_img,
                        fake_img_h,
                        fake_img_w,
                        fake_flow,
                        fake_flow_mask,
                    ) = self.generator(
                        mask,
                        feature_encoding,
                        edge_map if self.use_edge_map else None,
                        torch.cat(prior_fake_image_list, dim=1)
                        if self.num_prior_frames > 0
                        else None,
                        torch.cat(prior_mask_list, dim=1)
                        if self.num_prior_frames > 0
                        else None,
                    )

                    # Generate reference optical flow if needed
                    if self.use_optical_flow:
                        with torch_amp.autocast(enabled=False):
                            real_flow: torch.Tensor = self.flownet(
                                reference_image,
                                input_dict["img"][:, frame_no - 1].to(self.device),
                            ).detach()

                    # If using discriminators, calculate losses using them
                    if self.num_discriminators > 0:
                        assert (
                            fake_img.shape[1] == 1
                        ), "Discriminators not supported with multiple output images"

                        # Image discriminator
                        output_d_fake: torch.Tensor
                        output_d_fake, _ = self.image_discriminator(
                            (
                                mask,
                                edge_map if self.use_edge_map else None,
                                fake_img[:, 0].detach(),
                                *prior_fake_image_list,
                                *prior_mask_list,
                            )
                        )
                        loss_d_fake: torch.Tensor = (
                            self.image_discriminator.calculate_loss(
                                output_d_fake, fake_label, self.criterion_D
                            )
                        )

                        output_d_real: torch.Tensor
                        output_d_real, output_d_real_extra = self.image_discriminator(
                            (
                                mask,
                                edge_map if self.use_edge_map else None,
                                reference_image,
                                *prior_reference_image_list,
                                *prior_mask_list,
                            )
                        )
                        loss_d_real: torch.Tensor = (
                            self.image_discriminator.calculate_loss(
                                output_d_real, real_label, self.criterion_D
                            )
                        )

                        # Generator
                        output_g: torch.Tensor
                        output_g, output_g_extra = self.image_discriminator(
                            (
                                mask,
                                edge_map if self.use_edge_map else None,
                                fake_img[:, 0],
                                *prior_fake_image_list,
                                *prior_mask_list,
                            )
                        )
                        loss_g_gan: torch.Tensor = (
                            self.image_discriminator.calculate_loss(
                                output_g, real_label_gan, self.criterion_D
                            )
                        )

                        # Calculate feature matching loss for image discriminator
                        loss_g_fm: torch.Tensor = feature_matching_error(
                            output_d_real_extra,
                            output_g_extra,
                            self.feature_matching_weight,
                            self.num_discriminators,
                        )

                        # Sum losses together in manner that backward pass requires
                        loss_d_image = (loss_d_fake + loss_d_real) * 0.5
                        loss_g_image = loss_g_gan + loss_g_fm

                        # If using optical flow, use the extra discriminator
                        if self.use_optical_flow:
                            # Flow discriminator
                            output_d_fake_flow: torch.Tensor
                            output_d_fake_flow, _ = self.flow_discriminator(
                                (
                                    mask,
                                    edge_map if self.use_edge_map else None,
                                    fake_flow.detach(),
                                    *prior_mask_list,
                                    *prior_fake_flow_list,
                                )
                            )
                            loss_d_fake_flow: torch.Tensor = (
                                self.flow_discriminator.calculate_loss(
                                    output_d_fake_flow, fake_label, self.criterion_D
                                )
                            )

                            output_d_real_flow: torch.Tensor
                            (
                                output_d_real_flow,
                                output_d_real_extra_flow,
                            ) = self.flow_discriminator(
                                (
                                    mask,
                                    edge_map if self.use_edge_map else None,
                                    real_flow,
                                    *prior_mask_list,
                                    *prior_real_flow_list,
                                )
                            )
                            loss_d_real_flow: torch.Tensor = (
                                self.flow_discriminator.calculate_loss(
                                    output_d_real_flow, real_label, self.criterion_D
                                )
                            )

                            # Generator
                            output_g_flow: torch.Tensor
                            (
                                output_g_flow,
                                output_g_extra_flow,
                            ) = self.flow_discriminator(
                                (
                                    mask,
                                    edge_map if self.use_edge_map else None,
                                    fake_flow,
                                    *prior_mask_list,
                                    *prior_fake_flow_list,
                                )
                            )
                            loss_g_gan_flow: torch.Tensor = (
                                self.flow_discriminator.calculate_loss(
                                    output_g_flow, real_label_gan, self.criterion_D
                                )
                            )

                            loss_g_fm_flow: torch.Tensor = feature_matching_error(
                                output_d_real_extra_flow,
                                output_g_extra_flow,
                                self.feature_matching_weight,
                                self.num_discriminators,
                            )

                            # Prepare for backwards pass
                            loss_d_flow = (loss_d_fake_flow + loss_d_real_flow) * 0.5
                            loss_g_flow = loss_g_gan_flow + loss_g_fm_flow

                    if self.use_perceptual_loss:
                        # Calculate loss on final network output image
                        loss_img: torch.Tensor = self.loss_net(
                            fake_img, reference_image, mask
                        )
                        if self.use_optical_flow:
                            loss_img_h: torch.Tensor = self.loss_net(
                                fake_img_h.unsqueeze(1),  # Requires 5D tensor
                                reference_image,
                                mask,
                            )

                    if self.use_optical_flow:
                        # Warp prior reference image and compare to current reference image
                        warped_real_prev_image: torch.Tensor = FlowNetWrapper.resample(
                            prior_reference_image_list[0].detach(),
                            fake_flow,
                            self.generator.grid,
                        )
                        loss_warp_scaling_factor: float = 10.0
                        loss_warp: torch.Tensor = (
                            self.flow_criterion(
                                warped_real_prev_image, reference_image.detach()
                            )
                            * loss_warp_scaling_factor
                        )

                        # Calculate direct comparison optical flow loss
                        loss_flow_scaling_factor: float = 10.0
                        loss_flow: torch.Tensor = (
                            self.flow_criterion(fake_flow, real_flow)
                            * loss_flow_scaling_factor
                        )

                    # Previous frames stored for input later
                    if self.num_prior_frames > 0:

                        prior_fake_image_list = [
                            fake_img[:, 0].detach().clone().clamp(0.0, 1.0)
                            - (self.normalise_prior_frames * 0.5),
                            *prior_fake_image_list[0 : self.num_prior_frames - 1],
                        ]
                        prior_reference_image_list = [
                            reference_image.detach().clone()
                            - (self.normalise_prior_frames * 0.5),
                            *prior_reference_image_list[0 : self.num_prior_frames - 1],
                        ]
                        prior_mask_list = [
                            mask.detach().clone(),
                            *prior_mask_list[0 : self.num_prior_frames - 1],
                        ]
                        if self.use_optical_flow:
                            prior_real_flow_list = [
                                real_flow.detach(),
                                *prior_real_flow_list[0 : self.num_prior_frames - 2],
                            ]
                            prior_fake_flow_list = [
                                fake_flow.detach(),
                                *prior_fake_flow_list[0 : self.num_prior_frames - 2],
                            ]

                    # Add losses for CRNVideo together, no discriminator loss
                    loss: torch.Tensor = (
                        loss_img
                        + loss_img_h
                        + loss_warp
                        + loss_flow
                        + loss_g_image
                        + loss_g_flow
                    )

                # Do backwards passes, if using amp, do the fancy version
                # if self.use_amp == "torch":
                self.optimizer_G.zero_grad()
                self.torch_gradient_scaler.scale(loss).backward()
                self.torch_gradient_scaler.unscale_(self.optimizer_G)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 30)
                self.torch_gradient_scaler.step(self.optimizer_G)
                self.torch_gradient_scaler.update()

                if self.num_discriminators > 0:
                    self.optimizer_D_image.zero_grad()
                    self.torch_gradient_scaler.scale(loss_d_image).backward()
                    self.torch_gradient_scaler.unscale_(self.optimizer_D_image)
                    torch.nn.utils.clip_grad_norm_(
                        self.image_discriminator.parameters(), 30
                    )
                    self.torch_gradient_scaler.step(self.optimizer_D_image)
                    self.torch_gradient_scaler.update()

                    if self.use_optical_flow:
                        self.optimizer_D_flow.zero_grad()
                        self.torch_gradient_scaler.scale(loss_d_flow).backward()
                        self.torch_gradient_scaler.unscale_(self.optimizer_D_flow)
                        torch.nn.utils.clip_grad_norm_(
                            self.flow_discriminator.parameters(), 30
                        )
                        self.torch_gradient_scaler.step(self.optimizer_D_flow)
                        self.torch_gradient_scaler.update()

                # Loss scaler to scale to a per frame average
                loss_scaler: float = self.batch_size / num_frames

                # Extract losses from tensors
                loss_total += loss.item() * loss_scaler
                video_loss += loss.item() * loss_scaler

                video_loss_img += loss_img.item() * loss_scaler

                if self.use_optical_flow:
                    video_loss_img_h += loss_img_h.item() * loss_scaler
                    video_loss_flow += loss_flow.item() * loss_scaler
                    video_loss_warp += loss_warp.item() * loss_scaler

                if self.num_discriminators > 0:
                    video_loss_d += loss_d_image.item() * loss_scaler
                    video_loss_g_gan += loss_g_gan.item() * loss_scaler
                    video_loss_g_fm += loss_g_fm.item() * loss_scaler
                    video_output_d_real_mean += (
                        output_d_real.mean().item() * loss_scaler
                    )
                    video_output_d_fake_mean += (
                        output_d_fake.mean().item() * loss_scaler
                    )

                    if self.use_optical_flow:
                        video_loss_d_flow += loss_d_flow.item() * loss_scaler
                        video_loss_g_gan_flow += loss_g_gan_flow.item() * loss_scaler
                        video_loss_g_fm_flow += loss_g_fm_flow.item() * loss_scaler
                        video_output_d_real_mean_flow += (
                            output_d_real_flow.mean().item() * loss_scaler
                        )
                        video_output_d_fake_mean_flow += (
                            output_d_fake_flow.mean().item() * loss_scaler
                        )

            # Logging does not happen every single batch
            if log_this_batch:
                current_epoch: int = kwargs["current_epoch"]

                # Default logging includes perceptual loss
                wandb_log_dict: dict = {
                    "Epoch_Fraction": current_epoch
                    + ((batch_idx * self.batch_size) / len(self.dataset_train)),
                    "Batch Loss Total": video_loss,
                    "Batch Loss Final Image": video_loss_img,
                }

                if self.num_discriminators > 0:
                    wandb_log_dict.update(
                        {
                            "Batch Loss Discriminator Image": video_loss_d,
                            "Batch Loss Generator Image": video_loss_g_gan,
                            "Batch Loss Feature Matching Image": video_loss_g_fm,
                            "Output D Real Image": video_output_d_real_mean,
                            "Output D Fake Image": video_output_d_fake_mean,
                        }
                    )
                    if self.use_optical_flow:
                        wandb_log_dict.update(
                            {
                                "Batch Loss Discriminator Flow": video_loss_d_flow,
                                "Batch Loss Generator Flow": video_loss_g_gan_flow,
                                "Batch Loss Feature Matching Flow": video_loss_g_fm_flow,
                                "Output D Fake Flow": video_output_d_fake_mean_flow,
                                "Output D Real Flow": video_output_d_real_mean_flow,
                                "Batch Loss Hallucinated Image": video_loss_img_h,
                                "Batch Loss Warp": video_loss_warp,
                                "Batch Loss Flow": video_loss_flow,
                            }
                        )
                # Log the decided metrics
                wandb.log(wandb_log_dict)

        return loss_total, None

    def eval(self) -> Tuple[float, Any]:
        pass

    def sample(self, index: int, demovideo_dataset: bool = False) -> SampleDataHolder:

        if self.use_optical_flow:
            assert (
                self.num_frames_per_sampling_video > 1
            ), "self.use_optical_flow == True, but self.num_frames_per_sampling_video <= 1."

        num_frames_per_sampling_video = (
            self.num_frames_per_sampling_video
            + self.num_prior_frames
            + self.use_optical_flow
        )

        # Set generator to eval mode
        self.generator.eval()
        if self.use_feature_encodings:
            self.feature_encoder.eval()
        if self.num_discriminators > 0 and not self.sample_only:
            self.image_discriminator.eval()
            if self.use_optical_flow:
                self.flow_discriminator.eval()

        # Do not need gradients when sampling
        with torch.no_grad():

            transform: transforms.ToPILImage = transforms.ToPILImage()

            if not demovideo_dataset:
                input_dict = self.dataset_val[index]
            else:
                raise NotImplementedError("Demovideo support not added.")
                input_dict = self.dataset_demoVideo[index]

            reference_image_list: list = []
            mask_colour_list: list = []
            output_image_list: list = []
            feature_selection_list: list = []
            hallucinated_image_list: list = []
            warped_image_list: list = []
            combination_weights_list: list = []
            output_flow_list: list = []
            reference_flow_list: list = []

            mask_total = input_dict["msk"].unsqueeze(0)
            mask_colour_total = input_dict["msk_colour"].float().unsqueeze(0)
            instance_total = input_dict["inst"].unsqueeze(0)
            edge_map_total = input_dict["edge_map"].unsqueeze(0)
            real_img_total = input_dict["img"].unsqueeze(0)

            # No reference or optical flow history is kept
            prior_image_list: list = [
                torch.zeros_like(real_img_total[:, 0], device=self.device)
            ] * self.num_prior_frames
            prior_mask_list: list = [
                torch.zeros_like(mask_total[:, 0], device=self.device)
            ] * self.num_prior_frames

            num_frames_generated: int = 0

            # Loop through each frame
            for frame_no in range(self.use_optical_flow, num_frames_per_sampling_video):
                self.generator.zero_grad()

                # Extract single frame
                real_img: torch.Tensor = real_img_total[:, frame_no].to(self.device)
                mask: torch.Tensor = mask_total[:, frame_no].to(self.device)
                instance: torch.Tensor = instance_total[:, frame_no].to(self.device)
                edge_map: torch.Tensor = edge_map_total[:, frame_no].to(self.device)

                if self.use_feature_encodings:
                    if self.use_saved_feature_encodings:
                        feature_encoding: Optional[
                            torch.Tensor
                        ] = self.feature_encoder.sample_using_means(
                            instance, mask, fixed_class_lists=True
                        )
                    else:
                        feature_encoding = self.feature_encoder(
                            real_img, instance, mask=mask
                        )
                else:
                    feature_encoding = None

                # Setting output types
                fake_img: torch.Tensor
                fake_img_h: torch.Tensor
                fake_img_w: torch.Tensor
                fake_flow: torch.Tensor
                fake_flow_mask: torch.Tensor
                # Generate outputs from network
                # fake_img has format (batch x num_images x num_channels x height x width)
                (
                    fake_img,
                    fake_img_h,
                    fake_img_w,
                    fake_flow,
                    fake_flow_mask,
                ) = self.generator(
                    mask,
                    feature_encoding,
                    edge_map if self.use_edge_map else None,
                    torch.cat(prior_image_list, dim=1)
                    if self.num_prior_frames > 0
                    else None,
                    torch.cat(prior_mask_list, dim=1)
                    if self.num_prior_frames > 0
                    else None,
                )

                # Previous outputs stored for input later
                if self.num_prior_frames > 0:
                    prior_image_list = [
                        fake_img[:, 0].detach().clone().clamp(0.0, 1.0)
                        - (self.normalise_prior_frames * 0.5),
                        *prior_image_list[0 : self.num_prior_frames - 1],
                    ]
                    prior_mask_list = [
                        mask.detach(),
                        *prior_mask_list[0 : self.num_prior_frames - 1],
                    ]

                    if self.use_optical_flow:
                        # Reference flow for comparison to generated flow
                        real_flow: torch.Tensor = (
                            self.flownet(
                                real_img,
                                real_img_total[:, frame_no - 1].to(self.device),
                            )
                            .detach()
                            .permute(0, 2, 3, 1)
                        )

                        # Visualise flow for easier inspection
                        fake_flow_viz: torch.Tensor = (
                            torch.tensor(
                                fz.convert_from_flow(
                                    fake_flow.permute(0, 2, 3, 1)
                                    .squeeze()
                                    .cpu()
                                    .numpy()
                                )
                            )
                            .permute(2, 0, 1)
                            .float()
                            / 255.0
                        )

                        real_flow_viz: torch.Tensor = (
                            torch.tensor(
                                fz.convert_from_flow(real_flow.squeeze().cpu().numpy())
                            )
                            .permute(2, 0, 1)
                            .float()
                            / 255.0
                        )

                        if num_frames_generated >= self.num_prior_frames:
                            # Append transformed flow
                            hallucinated_image_list.append(
                                transform(fake_img_h.squeeze().clamp(0.0, 1.0).cpu())
                            )
                            warped_image_list.append(
                                transform(fake_img_w.squeeze().clamp(0.0, 1.0).cpu())
                            )
                            combination_weights_list.append(
                                transform(fake_flow_mask[0, 0].cpu())
                            )
                            output_flow_list.append(transform(fake_flow_viz))
                            reference_flow_list.append(transform(real_flow_viz))

                if num_frames_generated >= self.num_prior_frames:
                    reference_image_list.append(transform(real_img.squeeze().cpu()))
                    mask_colour_list.append(transform(mask_colour_total[0, frame_no]))

                    # Support for CRN 9 images
                    if self.num_frames_per_sampling_video > 1:
                        output_image_list.append(
                            transform(fake_img[0, 0].clamp(0.0, 1.0).cpu())
                        )
                    else:
                        for img_no in range(fake_img.shape[1]):
                            output_image_list.append(
                                transform(fake_img[0, img_no].clamp(0.0, 1.0).cpu())
                            )
                    if self.use_feature_encodings:
                        feature_selection_list.append(
                            transform(feature_encoding.squeeze().cpu())
                        )

                num_frames_generated += 1

            # Data holder struct for easy referencing of data
            output_data_holder: SampleDataHolder = SampleDataHolder(
                image_index=index,
                video_sample=self.num_frames_per_sampling_video > 1,
                reference_image=reference_image_list,
                mask_colour=mask_colour_list,
                output_image=output_image_list,
                hallucinated_image=hallucinated_image_list,
                warped_image=warped_image_list,
                combination_weights=combination_weights_list,
                output_flow=output_flow_list,
                reference_flow=reference_flow_list,
                feature_selection=feature_selection_list,
            )

            return output_data_holder

    def adjust_learning_rate(
        self,
        current_epoch: int,
        starting_epoch: int,
        ending_epoch: int,
        base_learning_rate: float,
    ) -> None:
        """
        Adjust an optimisers learning rate on a linear scale from the base learning rate down to zero.

        :param current_epoch: The current epoch in the training process.
        :param starting_epoch: Epoch that the learning rate adjustment should start at.
        :param ending_epoch: Epoch that the learning rate adjustment should end at.
        :param base_learning_rate: Initial learning rate the optimizer_crn is set to.
        :return: None.
        """

        # Return early if current epoch is less than the required starting epoch for lr degradation.
        if current_epoch < starting_epoch:
            return

        lr: float = base_learning_rate * (
            1.0
            - (
                (current_epoch - starting_epoch + 1)
                / (ending_epoch - starting_epoch + 2)
            )
        )
        print(
            "Changing learning rate:\n\tEpoch: {epoch}\n\tLearning rate: {lr}".format(
                epoch=current_epoch, lr=lr
            )
        )

        param_group: dict
        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr

        if self.num_discriminators > 0:
            for param_group in self.optimizer_D_image.param_groups:
                param_group["lr"] = lr

            if self.use_optical_flow:
                for param_group in self.optimizer_D_flow.param_groups:
                    param_group["lr"] = lr

    def enable_feature_encoder_learning(self):
        # Create params
        self.optimizer_G.add_param_group(
            {"params": self.generator.feature_encoder.parameters()}
        )

    def enable_global_generator_learning(self):
        # Create params
        self.optimizer_G.add_param_group(
            {"params": self.generator.global_generator.parameters()}
        )

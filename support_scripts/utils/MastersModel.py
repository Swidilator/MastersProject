import time
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Any, Optional

import torch

from support_scripts.sampling import SampleDataHolder
from support_scripts.utils import ModelSettingsManager


class MastersModel(ABC):
    @abstractmethod
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
        use_vid2vid_discriminators: bool,
        **kwargs,
    ):
        super(MastersModel, self).__init__()

        self.kwargs: dict = dict(kwargs)

        self.model: str = model
        self.model_name: Optional[str] = self.model

        self.device: torch.device = device
        self.dataset_path: str = dataset_path
        self.input_image_height_width: tuple = input_image_height_width
        self.batch_size: int = batch_size
        self.num_data_workers: int = num_data_workers
        self.training_subset_size: int = training_subset_size
        self.flip_training_images: bool = flip_training_images
        self.sample_only: bool = sample_only
        self.use_amp: Union[str, bool] = use_amp
        self.log_every_n_steps: int = log_every_n_steps
        self.model_save_dir: str = model_save_dir
        self.image_save_dir: str = image_save_dir
        self.starting_epoch: int = starting_epoch
        self.num_frames_per_training_video: int = num_frames_per_training_video
        self.num_frames_per_sampling_video: int = num_frames_per_sampling_video
        self.prior_frame_seed_type: str = prior_frame_seed_type
        self.use_mask_for_instances: bool = use_mask_for_instances
        self.video_frame_offset: Union[int, str] = video_frame_offset
        self.use_saved_feature_encodings: bool = use_saved_feature_encodings
        self.use_vid2vid_discriminators: bool = use_vid2vid_discriminators

    @property
    @abstractmethod
    def data_set_train(self) -> torch.utils.data.Dataset:
        pass

    @property
    @abstractmethod
    def data_set_val(self) -> torch.utils.data.Dataset:
        pass

    @property
    @abstractmethod
    def wandb_trainable_model(self) -> tuple:
        """
        Expose model components that WandB can track.

        :return: tuple containing model components that WandB can track.
        """
        pass

    @classmethod
    @abstractmethod
    def from_model_settings_manager(
        cls, manager: ModelSettingsManager
    ) -> "MastersModel":
        """
        Initialise MastersModel from ModelSettingsManager instead of input arguments.

        :param manager: ModelSettingsManager to sample settings from.
        :return: MastersModel
        """
        pass

    @classmethod
    @abstractmethod
    def load_model_with_embedded_settings(
        cls, manager: ModelSettingsManager
    ) -> "MastersModel":
        """
        Initialise MastersModel from a saved model using manager to find the model
        and using the model's saved arguments to initialise itself.

        :param manager: ModelSettingsManager to sample settings from.
        :return: MastersModel
        """
        pass

    @abstractmethod
    def __set_data_loader__(self, **kwargs) -> None:
        """
        Set up dataset and data loader for model.

        :param kwargs: Extra model specific settings for data loaders.
        :return: None
        """
        pass

    @abstractmethod
    def __set_model__(self, **kwargs) -> None:
        """
        Set up all necessary model components into a state ready for training.

        :param kwargs: Extra model specific settings for model creation.
        :return: None
        """
        pass

    def __get_model_snapshot_name__(self) -> str:
        """
        Generate a timestamped name string for saving a model.

        :return: str
        """
        """
        Generate a timestamped name string for saving a model.

        Returns:
            Snapshot model name
        """
        assert self.model_name is not None

        localtime: time.localtime() = time.localtime(time.time())
        model_snapshot: str = self.model_name + "_"
        model_snapshot = model_snapshot + str(localtime.tm_year) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_mon) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_mday) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_hour) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_min) + "_"
        model_snapshot = model_snapshot + str(localtime.tm_sec) + ".pt"

        return model_snapshot

    @abstractmethod
    def save_model(self, epoch: int = -1) -> None:
        """
        Save the current model state. Defaults to only overwriting last file.

        Args:
            epoch: Save an extra version every epoch

        Returns:
            None
        """
        assert self.model_name is not None

    @abstractmethod
    def load_model(
        self,
        model_file_name: str,
    ) -> None:
        """
        Inplace load a snapshot of a model.

        Args:
            model_file_name: Filename of saved model file

        Returns:
            None
        """
        assert self.model_name is not None

    @abstractmethod
    def train(self, **kwargs) -> Tuple[float, Any]:
        """
        Perform one training epoch.

        Args:
            kwargs: Model and epoch specific settings

        Returns:
            Tuple[float, Any]
        """
        pass

    @abstractmethod
    def eval(self) -> Tuple[float, Any]:
        """
        Evaluate model on test_dataset.

        Returns:
            Tuple[float, Any]
        """
        pass

    @abstractmethod
    def sample(self, index: int, demovideo_dataset: bool = False) -> SampleDataHolder:
        """
        Sample k random images from dataset, forward through network.

        Args:
            index: image index to sample
            demovideo_dataset: use demovideo dataset instead.

        Returns:
            Union[dict, List[dict]]
        """
        pass

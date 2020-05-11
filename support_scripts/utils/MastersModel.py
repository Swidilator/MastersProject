import torch
import time
from typing import Union, Tuple, List, Any
from support_scripts.utils import ModelSettingsManager
import torch.utils as utils

from abc import ABC, abstractmethod


class MastersModel(ABC):
    @abstractmethod
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        input_image_height_width: tuple,
        batch_size_slice: int,
        batch_size_total: int,
        num_classes: int,
        num_loader_workers: int,
        subset_size: int,
        should_flip_train: bool,
        use_tanh: bool,
        use_input_noise: bool,
        sample_only: bool,
        **kwargs,
    ):
        super(MastersModel, self).__init__()

        self.model_name: Union[str, None] = None

        self.device: torch.device = device
        self.data_path: str = data_path
        self.input_image_height_width: tuple = input_image_height_width
        self.batch_size_slice: int = batch_size_slice
        self.batch_size_total: int = batch_size_total
        self.num_classes: int = num_classes
        self.num_loader_workers: int = num_loader_workers
        self.subset_size: int = subset_size
        self.should_flip_train: bool = should_flip_train
        self.use_tanh: bool = use_tanh
        self.use_input_noise: bool = use_input_noise
        self.sample_only: bool = sample_only

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
    def save_model(self, model_dir: str, epoch: int = -1) -> None:
        """
        Save the current model state. Defaults to only overwriting last file.

        Args:
            model_dir: Directory in which to save the model
            epoch: Save an extra version every epoch

        Returns:
            None
        """
        assert self.model_name is not None

    @abstractmethod
    def load_model(self, model_dir: str, model_file_name: str) -> None:
        """
        Inplace load a snapshot of a model.

        Args:
            model_dir: Path to saved model folder
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
    def sample(self, k: int, **kwargs) -> List[Tuple[Any, Any]]:
        """
        Sample k random images from dataset, forward through network.

        Args:
            k: number of images to sample
            kwargs: any model-specific arguments

        Returns:
            List[Tuple[Any, Any]]
        """
        pass

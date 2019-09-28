import torch
import time
from typing import Union
import torch.utils as utils

from Helper_Stuff import epoch_output, sample_output
from abc import ABC, abstractmethod


class MastersModel(ABC):
    @abstractmethod
    def __init__(
        self,
        device: torch.device,
        data_path: str,
        batch_size_slice: int,
        batch_size_total: int,
        num_loader_workers: int,
        subset_size: int,
        settings: dict,
    ):
        super(MastersModel, self).__init__()

        self.model_name: Union[str, None] = None

        self.device: torch.device = device
        self.data_path: str = data_path
        self.batch_size_slice: int = batch_size_slice
        self.batch_size_total: int = batch_size_total
        self.num_loader_workers: int = num_loader_workers
        self.subset_size: int = subset_size
        self.settings: dict = settings

    @property
    @abstractmethod
    def wandb_trainable_model(self) -> tuple:
        """
        Give access to the list of models that WandB must track

        Returns:
            Models to track
        """
        pass

    @abstractmethod
    def __set_data_loader__(
        self,
        data_path: str,
        batch_size_total: int,
        num_loader_workers: int,
        subset_size: int,
        settings: dict,
    ) -> None:
        """
        Set up dataset and data loader for model.

        Args:
            data_path: Path to dataset
            batch_size: Batch size for data-loaders
            num_loader_workers: Number of CPU workers for loading data
            subset_size: Maximum number of images to load in training set
            settings: Extra model-specific settings

        Returns:
            None
        """
        pass

    @abstractmethod
    def __set_model__(self, settings: dict) -> None:
        """
        Set up all necessary model components into a state ready for training.

        Args:
            settings: Model-specific settings

        Returns:
            None
        """
        pass

    def __get_model_snapshot_name__(self) -> str:
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
    def save_model(self, model_dir: str, snapshot: bool = False) -> None:
        """
        Save the current model state. Defaults to only overwriting last file.

        Args:
            model_dir: Directory in which to save the model
            snapshot: Save an extra, timestamped version

        Returns:
            None
        """
        assert self.model_name is not None

    @abstractmethod
    def load_model(self, model_dir: str, model_name: str) -> None:
        """
        Inplace load a snapshot of a model.

        Args:
            model_dir: Path to saved model folder
            model_name: Filename of saved model file

        Returns:
            None
        """
        assert self.model_name is not None

    @abstractmethod
    def train(self, input: tuple) -> epoch_output:
        """
        Perform one training epoch.

        Args:
            input: Model and epoch specific settings

        Returns:
            epoch_output
        """
        pass

    @abstractmethod
    def eval(self) -> epoch_output:
        """
        Evaluate model on test_dataset.

        Returns:
            epoch_output
        """
        pass

    @abstractmethod
    def sample(self, k: int) -> sample_output:
        """
        Sample k random images from dataset, forward through network.

        Args:
            k: number of images to sample

        Returns:
            sample_output
        """
        pass

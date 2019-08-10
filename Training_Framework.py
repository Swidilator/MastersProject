import torch
import torch.utils as utils

from Helper_Stuff import epoch_output, sample_output
from abc import ABC, abstractmethod


class MastersModel(ABC):
    def __init__(self, device: torch.device):
        super(MastersModel, self).__init__()

        self.device: torch.device = device
        self.data_loader: utils.data.DataLoader = None

    @abstractmethod
    def __set_data_loader__(self, data_path: str) -> None:
        r"""
        Set up dataset and data loader for model.
        :param data_path: Path to dataset
        :return: None
        """
        pass

    @abstractmethod
    def __set_model__(self) -> None:
        r"""
        Set up all necessary model components into a state ready for training.
        :return: None
        """
        pass

    @abstractmethod
    def save_model(self, model_dir: str) -> None:
        r"""
        Save a snapshot of the model.
        :param model_dir: Path to saved model folder
        :return: None
        """
        pass

    @abstractmethod
    def load_model(self, model_dir: str, model_name: str) -> None:
        r"""
        Inplace load a snapshot of a model.
        :param model_dir: Path to saved model folder
        :param model_name: Filename of saved model file
        :return: None
        """
        pass

    @abstractmethod
    def train(self) -> epoch_output:
        r"""
        Perform one training epoch.
        :return: epoch_output
        """
        pass

    @abstractmethod
    def eval(self) -> epoch_output:
        r"""
        Evaluate model on test_dataset
        :return: epoch_output
        """
        pass

    @abstractmethod
    def sample(self, k: int) -> sample_output:
        r"""
        Sample k random images from dataset, forward through network
        :return: sample_output
        """
        pass

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.modules as modules
import torchvision

from typing import List
from abc import ABC, abstractmethod

train_output = List[float, torch.Tensor]


class MastersModel(ABC):
    def __init__(self, device: torch.device, data_path: str):
        super(MastersModel, self).__init__()

        self.device: torch.device = device
        self.data_loader: utils.data.DataLoader = None

        self.__set_data_loader__(data_path)
        self.__set_model__()

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
    def train(self) -> train_output:
        r"""
        Perform one training epoch.
        :return: train_output
        """
        pass

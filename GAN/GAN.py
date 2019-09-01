import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.modules as modules
from torch.nn.functional import one_hot
from math import log2
import time
import random

from Helper_Stuff import *
from Data_Management import CRNDataset
from CRN.Perceptual_Loss import PerceptualLossNetwork
from Training_Framework import MastersModel

import wandb


class GAN(torch.nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

    def forward(self):
        pass

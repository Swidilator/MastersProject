import numpy as np
import torchvision
import torch
from matplotlib import pyplot as plt

from PerceptualLoss import PerceptualDifference, PerceptualLossNetwork

from CRN import RefinementModule

if __name__ == "__main__":
    rm = RefinementModule(61, 64, 3, torch.Size([20, 3, 64, 64]), False)
    # noinspection PyArgumentList
    input_test = torch.FloatTensor(20, 64, 64, 64).uniform_(0, 255)
    output_test = rm(input_test)

    plt.figure(0)
    plt.imshow(input_test[0, 0, :].numpy())
    plt.show()
    plt.figure(1)
    plt.imshow(output_test[0, 0, :].detach().numpy())
    plt.show()

    loss_holder: PerceptualLossNetwork = PerceptualLossNetwork()
    loss = loss_holder.layer_list[0](output_test) - loss_holder.layer_list[0](input_test)
    loss_sum = loss.sum()
    loss_sum.backward()


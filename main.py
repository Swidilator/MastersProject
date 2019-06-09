import numpy as np
import torchvision
import torch
from matplotlib import pyplot as plt

from CRN import RefinementModule

if __name__ == "__main__":
    rm = RefinementModule(3, 10, 3, torch.Size([20, 3, 32, 32]), False)
    input_test = torch.FloatTensor(4, 6, 32, 32).uniform_(0, 255)
    output_test = rm(input_test)

    plt.figure(0)
    plt.imshow(input_test[0, 0, :].numpy())
    plt.show()
    plt.figure(1)
    plt.imshow(output_test[0, 0, :].detach().numpy())
    plt.show()


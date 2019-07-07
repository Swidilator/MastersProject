import torchvision
import torch
from matplotlib import pyplot as plt
from Data_Management import CRNDataset

from CRN import RefinementModule

if __name__ == "__main__":

    # data_set = torchvision.datasets.Cityscapes(root="C:\CityScapes Samples\Train",
    #                                            split="train",
    #                                            mode="fine",
    #                                            target_type="semantic",
    #                                            # target_transform=transforms
    #                                            )
    data_set = CRNDataset(
        max_input_height_width=(32, 64),
        root="C:\\CityScapes Samples\\Train\\",
        split="train",
    )
    a, b = data_set[0]

    # data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    #     data_set, batch_size=1, shuffle=True, num_workers=1
    # )

    # image, smnt = next(iter(data_loader))
    # print(type(image))

    # plt.imshow(torchvision.transforms.ToPILImage(image[0].detach().numpy()))
    # plt.show()

    # rm = RefinementModule(
    #     prior_layer_channel_count=0,
    #     semantic_input_channel_count=3,
    #     output_channel_count=3,
    #     input_height_width=(32, 64),
    #     is_final_module=False,
    # )

    # output_test = rm(smnt[5])

    # plt.figure(0)
    # plt.imshow(torchvision.transforms.ToPILImage(smnt[0]))
    # plt.show()
    # plt.figure(1)
    # plt.imshow(output_test[0].detach().numpy())
    # plt.show()


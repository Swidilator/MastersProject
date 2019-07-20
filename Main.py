import torchvision
import torch
from matplotlib import pyplot as plt
from Data_Management import CRNDataset
from CRN import CRN

from CRN import RefinementModule

if __name__ == "__main__":
    max_input_height_width: tuple = (16, 32)

    data_set_standard = torchvision.datasets.Cityscapes(
        root="../CityScapes Samples/Train/", split="train", mode="fine", target_type="semantic"
    )

    data_set = CRNDataset(
        max_input_height_width=max_input_height_width,
        root="../CityScapes Samples/Train/",
        split="train",
        num_classes=35
    )
    a, b = data_set[0]
    print(a.shape, b.shape)

    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        data_set, batch_size=1, shuffle=True, num_workers=2
    )

    image, smnt = next(iter(data_loader))
    print(smnt.shape)

    crn = CRN(
        final_image_size=max_input_height_width,
        num_output_images=3,
        num_classes=35
    )

    out = crn(smnt)
    to_image = torchvision.transforms.ToPILImage()
    plt.figure(0)
    plt.imshow(to_image(out[0, 3:6]))
    plt.show()

    # plt.imshow(torchvision.transforms.ToPILImage(image[0].detach().numpy()))
    # plt.show()

    # rm = RefinementModule(
    #     prior_layer_channel_count=0,
    #     semantic_input_channel_count=35,
    #     output_channel_count=3,
    #     input_height_width=(32, 64),
    #     is_final_module=False,
    # )

    # output_test = rm(smnt, prior_layers=None)

    # plt.figure(0)
    # plt.imshow(torchvision.transforms.ToPILImage(smnt[0]))
    # plt.show()
    # plt.figure(1)
    # plt.imshow(output_test[0].detach().numpy())
    # plt.show()


import torchvision
import torch
from matplotlib import pyplot as plt
from Data_Management import CRNDataset
from CRN import CRN

from CRN import RefinementModule

if __name__ == "__main__":
    MAX_INPUT_HEIGHT_WIDTH: tuple = (128, 256)
    INPUT_TENSOR_SIZE: tuple = (4, 8)
    NUM_OUTPUT_IMAGES: int = 1
    NUM_CLASSES: int = 35
    BATCH_SIZE: int = 1
    PREFER_CUDA: bool = True

    if PREFER_CUDA == 1:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Device: CUDA')
        else:
            device = torch.device("cpu")
            print('Device: CPU')
    else:
        device = torch.device("cpu")
        print('Device: CPU')

    data_set_standard = torchvision.datasets.Cityscapes(
        root="../CityScapes Samples/Train/", split="train", mode="fine", target_type="semantic"
    )

    data_set = CRNDataset(
        max_input_height_width=MAX_INPUT_HEIGHT_WIDTH,
        root="../CityScapes Samples/Train/",
        split="train",
        num_classes=NUM_CLASSES
    )
    a, b = data_set[0]
    print(a.shape, b.shape)

    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    image, msk = next(iter(data_loader))
    print(msk.shape)

    crn = CRN(
        input_tensor_size=INPUT_TENSOR_SIZE,
        final_image_size=MAX_INPUT_HEIGHT_WIDTH,
        num_output_images=NUM_OUTPUT_IMAGES,
        num_classes=NUM_CLASSES
    )

    noise: torch.Tensor = torch.randn(BATCH_SIZE, 1, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], device=device)

    crn = crn.to(device)
    msk = msk.to(device)
    noise.to(device)

    out = crn(
        mask=msk,
        noise=noise,
        batch_size=BATCH_SIZE
    )
    out.sum().backward()
    to_image = torchvision.transforms.ToPILImage()
    plt.figure(0)
    plt.imshow(to_image(out[0, 0:3].cpu()))
    plt.show()
    del out, crn, msk, noise
    torch.cuda.empty_cache()
    quit()

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

